"""
GRPO 强化学习训练脚本

基于 trl + peft + DeepSeek API 奖励模型，对 SFT 微调后的模型进行 GRPO 对齐训练。

原理:
  GRPO (Group Relative Policy Optimization): 对同一个 prompt 生成多个候选，
  组内相对比较，好的被强化，差的被抑制。不需要 Value Network，单卡即可运行。

用法:
    python -m src.training.grpo_train \
      --model /path/to/merged/model \
      --lora /path/to/sft/lora \
      --episodes 500 \
      --output_dir output/grpo

硬件: 1× RTX 6000D 80GB (¥5.18/h)
预估: 500 episodes × 2 candidates × ~45s = 6-8 小时, ¥35-45
"""

import argparse, asyncio, csv, gc, hashlib, json, os, random, re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import httpx
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# ============================================================
# 配置
# ============================================================

# .env 加载
_env = Path(__file__).resolve().parent.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@dataclass
class GRPOConfig:
    # 模型
    model_path: str = "models/novel-merged"
    sft_lora_path: Optional[str] = None  # SFT 的 LoRA 作为初始化

    # GRPO
    group_size: int = 2                  # 每个 prompt 生成候选数
    kl_penalty: float = 0.05            # KL 散度惩罚系数
    learning_rate: float = 1e-5          # RL 学习率（比 SFT 低 20 倍）
    max_new_tokens: int = 2048           # 每次生成长度
    temperature: float = 0.8
    top_p: float = 0.9
    total_episodes: int = 500
    max_prompt_length: int = 4096

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0           # MoE 限制

    # 训练控制
    gradient_accumulation_steps: int = 2   # 梯度累积步数。防御: 训练启动时自动确保 >= group_size
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3

    # 数据
    train_file: str = "data/sft/train.jsonl"
    num_prompts: int = 500               # 训练 prompt 数量

    # 输出
    output_dir: str = "output/grpo"
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100

    # API
    max_concurrent: int = 5              # DeepSeek API 最大并发请求数


# ============================================================
# 奖励模型（DeepSeek API）
# ============================================================

REWARD_PROMPT = """你是网文质量评审专家。请对以下都市文娱小说片段打分。

评分维度 (每项 1-10):
1. 文风匹配度: 是否轻松诙谐、口语化、有梗
2. 爽感: 是否有打脸/逆袭/反转的爽感
3. 结构: 字数适中、起承转合、结尾有钩子
4. 连贯性: 情节自然、人物一致
5. 都市文娱专业感: 娱乐圈/影视行业描写是否真实

【章节内容】
%s

请按 JSON 格式输出:
{"文风匹配度": X, "爽感": X, "结构": X, "连贯性": X, "专业感": X, "总评": "一句话"}"""


class RewardModel:
    """基于 DeepSeek API 的奖励模型"""

    def __init__(self, model: str = "deepseek-v4-flash", max_concurrent: int = 5):
        self.model = model
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self._cache = {}
        self._max_concurrent = max_concurrent  # DeepSeek API 并发上限

    async def score_batch_async(self, candidates: list[str], context: dict = None) -> list[float]:
        """真正异步批量打分 — httpx.AsyncClient + Semaphore 控制并发"""
        sem = asyncio.Semaphore(self._max_concurrent)
        async with httpx.AsyncClient(timeout=30) as client:
            async def _score_one(text: str) -> float:
                async with sem:
                    return await self._score_async(text, context, client)
            return await asyncio.gather(*[_score_one(c) for c in candidates])

    def score_batch(self, candidates: list[str], context: dict = None) -> list[float]:
        """同步批量打分（自动选择最优路径）"""
        if not self.api_key:
            return [self.score(c, context) for c in candidates]
        # Python 3.12 兼容: 用 asyncio.run() 替代废弃的 get_event_loop()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 无 running loop → 直接用 asyncio.run（标准路径）
            return asyncio.run(self.score_batch_async(candidates, context))
        # 已有 running loop（如 Jupyter）→ 线程池并行回退
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self._max_concurrent, len(candidates))) as ex:
            return list(ex.map(lambda c: self.score(c, context), candidates))

    def score(self, chapter: str, context: dict = None) -> float:
        """5 维加权总分，与项目规划文档完全对齐"""
        # 1. LLM 打分（文风 + 爽感 + 专业感 + 连贯性）
        llm = self._llm_score(chapter)

        # 2. 结构分（字数 + 高潮检测 + 钩子检测）
        wc = len(chapter.replace("\n", "").replace(" ", ""))
        wc_score = (1.0 if 2000 <= wc <= 3000 else 0.5 if abs(wc - 2500) < 1000 else 0.0)
        climax_score = 0.8 if self._has_climax(chapter) else 0.2
        hook_score = 0.8 if self._has_hook(chapter) else 0.2
        structure = wc_score * 0.4 + climax_score * 0.3 + hook_score * 0.3

        # 3. 重复度检查
        diversity = 1.0
        if context and context.get("previous"):
            prev_texts = [t for t in context["previous"] if t]
            if prev_texts:
                diversity = 1.0 - max(
                    self._ngram_overlap(chapter[:500], ch[:500])
                    for ch in prev_texts
                )

        return (
            0.25 * llm.get("文风匹配度", 3) / 10 +
            0.25 * structure +
            0.20 * llm.get("连贯性", 3) / 10 +
            0.20 * diversity +
            0.10 * (llm.get("爽感", 3) + llm.get("专业感", 3)) / 20  # 都市文娱专项=爽感+专业感
        )

    def _llm_score(self, text: str) -> dict:
        """调用 DeepSeek API 打分"""
        # 缓存命中（用文本前 500 字的 hash 作为 key，避免前 100 字碰撞）
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.api_key:
            return {"文风匹配度": 5, "爽感": 5, "连贯性": 5, "专业感": 5}

        try:
            resp = httpx.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": REWARD_PROMPT % text[:3000]}],
                    "max_tokens": 256, "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            scores = json.loads(resp.json()["choices"][0]["message"]["content"])
            self._cache[cache_key] = scores
            return scores
        except Exception:
            return {"文风匹配度": 5, "爽感": 5, "连贯性": 5, "专业感": 5}

    # ================================================================
    # 真正异步方法（httpx.AsyncClient + asyncio）
    # ================================================================

    async def _llm_score_async(self, text: str, client: httpx.AsyncClient) -> dict:
        """异步调用 DeepSeek API 打分（与 _llm_score 逻辑一致，仅 IO 层不同）"""
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        if not self.api_key:
            return {"文风匹配度": 5, "爽感": 5, "连贯性": 5, "专业感": 5}
        try:
            resp = await client.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": REWARD_PROMPT % text[:3000]}],
                    "max_tokens": 256, "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            scores = json.loads(resp.json()["choices"][0]["message"]["content"])
            self._cache[cache_key] = scores
            return scores
        except Exception:
            return {"文风匹配度": 5, "爽感": 5, "连贯性": 5, "专业感": 5}

    async def _score_async(self, chapter: str, context: dict, client: httpx.AsyncClient) -> float:
        """异步版单条打分（与 score 逻辑一致）"""
        llm = await self._llm_score_async(chapter, client)
        wc = len(chapter.replace("\n", "").replace(" ", ""))
        wc_score = (1.0 if 2000 <= wc <= 3000 else 0.5 if abs(wc - 2500) < 1000 else 0.0)
        climax_score = 0.8 if self._has_climax(chapter) else 0.2
        hook_score = 0.8 if self._has_hook(chapter) else 0.2
        structure = wc_score * 0.4 + climax_score * 0.3 + hook_score * 0.3
        diversity = 1.0
        if context and context.get("previous"):
            prev_texts = [t for t in context["previous"] if t]
            if prev_texts:
                diversity = 1.0 - max(
                    self._ngram_overlap(chapter[:500], ch[:500]) for ch in prev_texts
                )
        return (
            0.25 * llm.get("文风匹配度", 3) / 10 +
            0.25 * structure +
            0.20 * llm.get("连贯性", 3) / 10 +
            0.20 * diversity +
            0.10 * (llm.get("爽感", 3) + llm.get("专业感", 3)) / 20
        )

    @staticmethod
    def _ngram_overlap(a: str, b: str) -> float:
        """4-gram 重叠率"""
        a_clean = re.sub(r"\s", "", a)
        b_clean = re.sub(r"\s", "", b)
        if len(a_clean) < 4 or len(b_clean) < 4:
            return 0
        a_ngrams = {a_clean[i:i+4] for i in range(len(a_clean)-3)}
        b_ngrams = {b_clean[i:i+4] for i in range(len(b_clean)-3)}
        if not b_ngrams:
            return 0
        return len(a_ngrams & b_ngrams) / len(b_ngrams)

    @staticmethod
    def _has_climax(text: str) -> bool:
        """检测文本中是否有高潮/爽点（关键词匹配）"""
        climax_keywords = [
            "打脸", "反转", "爆发", "震惊", "沸腾", "掌声", "获奖",
            "破纪录", "夺冠", "碾压", "逆袭", "封神", "全场", "欢呼"
        ]
        return any(kw in text for kw in climax_keywords)

    @staticmethod
    def _has_hook(text: str) -> bool:
        """检测结尾是否留钩子（最后 3 句检查）"""
        sentences = re.split(r"[。！？\n]", text)
        last_3 = "".join(sentences[-4:]) if len(sentences) >= 4 else text[-200:]
        hook_patterns = [r"[？?]", r"但是|然而|不料|没想到|谁知",
                         r"接下来|明天|等.*再|看.*怎么", r"…|\.{3}|——"]
        return any(re.search(p, last_3) for p in hook_patterns)


# ============================================================
# Prompt 池
# ============================================================

class PromptPool:
    """训练 prompt 池"""

    def __init__(self, config: GRPOConfig):
        self.prompts = []
        self._load_prompts(config)

    def _load_prompts(self, config: GRPOConfig):
        """混合加载: SFT 数据 + 大纲约束系统"""
        # 来源 1: SFT 训练数据
        sft_path = Path(config.train_file)
        if sft_path.exists():
            sft_prompts = []
            with open(sft_path) as f:
                for line in f:
                    item = json.loads(line)
                    user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
                    if user_msg:
                        sft_prompts.append(user_msg)
            random.shuffle(sft_prompts)
            self.prompts.extend(sft_prompts[:config.num_prompts // 2])

        # 来源 2: 大纲系统生成的指令（本地模拟，不需要 API）
        outline_prompts = []
        templates = [
            "重生2008年，{}。续写本章，2000-3000字。",
            "{}这是本章高潮，请写出爽感。",
            "{}请写出打脸场面，要爽。",
            "{}请写一段都市文娱日常。",
        ]
        settings = [
            "北电导演系学生林风绑定文娱系统",
            "林风的新电影即将上映",
            "投资人质疑林风的能力",
            "综艺录制现场出现意外",
            "林风的前女友突然出现",
        ]
        for _ in range(config.num_prompts // 2):
            t = random.choice(templates)
            s = random.choice(settings)
            outline_prompts.append(t.format(s + "。"))
        self.prompts.extend(outline_prompts)

        random.shuffle(self.prompts)
        print(f"📋 Prompt 池: {len(self.prompts)} 条 (SFT采样 + 模板生成)")

    def sample(self, n: int = 1) -> list[str]:
        return random.sample(self.prompts, min(n, len(self.prompts)))


# ============================================================
# 训练循环
# ============================================================

class GRPOTrainer:
    """GRPO 训练器"""

    def __init__(self, config: GRPOConfig, resume: bool = False):
        self.config = config
        self.reward_model = RewardModel(max_concurrent=config.max_concurrent)
        self.prompt_pool = PromptPool(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()
        self._setup_optimizer()

        # 尝试从 checkpoint 恢复训练
        self.train_rewards = []
        self.best_avg_reward = 0.0
        self.no_improve_count = 0
        self._start_episode = 1
        if resume:
            self._load_state()

    def _setup_model(self):
        """加载模型 + LoRA"""
        print(f"🤖 加载模型: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True, device_map="auto"
        )

        # 加载 SFT LoRA 作为初始化（合并进基座后重建 rank=32 GRPO LoRA）
        grpo_lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank, lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj","v_proj","k_proj","o_proj","gate_up_proj","down_proj"],
        )
        if self.config.sft_lora_path and Path(self.config.sft_lora_path).exists():
            print(f"📦 加载 SFT LoRA: {self.config.sft_lora_path}")
            try:
                self.model = PeftModel.from_pretrained(self.model, self.config.sft_lora_path)
                # 将 SFT LoRA 合并进基座权重，保留 SFT 知识
                self.model = self.model.merge_and_unload()
            except TypeError:
                from peft import PeftModel as PM
                self.model = PM.from_pretrained(self.model, self.config.sft_lora_path)
                self.model = self.model.merge_and_unload()
            # 统一新建 GRPO LoRA (rank=32)，确保显存预算与规划一致
            print(f"🔧 新建 GRPO LoRA: rank={self.config.lora_rank}, alpha={self.config.lora_alpha}")
            self.model = get_peft_model(self.model, grpo_lora_config)
        else:
            print(f"🔧 新建 GRPO LoRA (无 SFT 初始化): rank={self.config.lora_rank}")
            self.model = get_peft_model(self.model, grpo_lora_config)

        self.model.train()
        # 冻结基座，只训练 LoRA
        for n, p in self.model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False

    def _setup_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.config.learning_rate)
        print(f"🔧 可训练参数: {sum(p.numel() for p in trainable):,}")

    # ================================================================
    # 训练主循环
    # ================================================================

    def train(self):
        config = self.config

        # 防御: gradient_accumulation_steps 必须 >= group_size
        if config.gradient_accumulation_steps < config.group_size:
            print(f"⚠️ gradient_accumulation_steps ({config.gradient_accumulation_steps}) < "
                  f"group_size ({config.group_size}), 自动调整为 {config.group_size}")
            config.gradient_accumulation_steps = config.group_size

        # wandb 记录训练曲线（登录失败则跳过）
        self._use_wandb = False
        try:
            run_name = f"grpo-g{config.group_size}-r{config.lora_rank}"
            wandb.init(project="novel-llm", name=run_name, config={
                "group_size": config.group_size, "kl_penalty": config.kl_penalty,
                "learning_rate": config.learning_rate, "total_episodes": config.total_episodes,
            })
            self._use_wandb = True
        except Exception as e:
            print(f"⚠️ wandb 不可用: {e}")

        print(f"\n🚀 GRPO 训练开始: {config.total_episodes} episodes\n")

        prev_texts = []  # 用于重复度检查

        pbar = tqdm(range(self._start_episode, config.total_episodes + 1),
                    desc="GRPO", unit="ep", ncols=100)
        for episode in pbar:
            prompt = self.prompt_pool.sample(1)[0]
            rewards = []
            candidates = []

            # 1. 批量生成 group_size 个候选（单次 forward = GPU 满载，no_grad 已内置）
            candidates = self._generate_batch(prompt, config.group_size, seed=episode)

            # 1.5 释放生成的 KV cache，为 loss 前向留显存
            torch.cuda.empty_cache()

            # 2. 真正异步批量打分 —
            #    并发调用 DeepSeek API，将等待时间从 N×30s 压缩到 ~30s
            context_for_reward = {"previous": prev_texts[-3:]} if prev_texts else None
            rewards = self.reward_model.score_batch(candidates, context_for_reward)

            # 3. GRPO: 组内相对比较
            rewards_tensor = torch.tensor(rewards, device=self.device)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_r) / std_r  # z-score 归一化

            # 4. 对每个候选计算 loss 并更新
            #    注: GRPO 的 on-policy 特性要求生成后立即更新。
            #    真正的 GPU 利用率提升靠增大 group_size（更多候选=更多 GPU 工作时间占比）
            total_loss = 0.0
            grad_steps = 0

            for i in range(config.group_size):
                weight = max(advantages[i].item(), 0.05)
                loss = self._compute_grpo_loss(candidates[i], prompt, weight)
                (loss / config.gradient_accumulation_steps).backward()
                total_loss += loss.item()
                grad_steps += 1
                del loss  # 立即释放 loss 持有的计算图

                if grad_steps % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # 4. 每 episode 结束时 flush 残留梯度
            if grad_steps % config.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 记录
            self.train_rewards.append(mean_r.item())
            prev_texts.append(candidates[rewards.index(max(rewards))])  # 保存 best

            # 计算移动平均
            avg_r = sum(self.train_rewards[-50:]) / min(len(self.train_rewards), 50)

            # wandb 记录
            wandb.log({
                "episode": episode,
                "reward_mean": mean_r.item(),
                "reward_std": std_r.item(),
                "reward_max": rewards_tensor.max().item(),
                "loss": total_loss / max(grad_steps, 1),
                "avg_reward_50": avg_r,
            })

            # 更新进度条
            pbar.set_postfix({                         # Use the pbar variable from the outer scope
                "avg_r": f"{avg_r:.3f}",
                "r": f"{mean_r.item():.3f}",
                "loss": f"{total_loss/max(grad_steps,1):.4f}",
                "best": f"{self.best_avg_reward:.3f}",
            })

            # 日志（用 tqdm.write 避免破坏进度条）
            if episode % config.log_interval == 0:
                pbar.write(f"  [{episode:4d}/{config.total_episodes}] "
                           f"rewards={[f'{r:.3f}' for r in rewards]}")

            # 评估
            if episode % config.eval_interval == 0:
                self._evaluate(episode)

            # 保存
            if episode % config.save_interval == 0:
                avg_r = sum(self.train_rewards[-50:]) / min(len(self.train_rewards), 50)
                # 总是保存 latest（用于恢复训练）
                self._save("latest")
                # reward 提升时额外保存 best
                if avg_r > self.best_avg_reward:
                    self.best_avg_reward = avg_r
                    self.no_improve_count = 0
                    self._save("best")
                else:
                    self.no_improve_count += 1
                if self.no_improve_count >= config.early_stopping_patience:
                    print(f"\n⏹️ Early stopping: reward 连续 {config.early_stopping_patience} 轮无提升")
                    break

            # 释放
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 最终保存
        self._save("final")
        wandb.finish()
        print(f"\n✅ GRPO 训练完成! best_avg_reward={self.best_avg_reward:.4f}")

    # ================================================================
    # 内部方法
    # ================================================================

    def _generate(self, prompt: str, seed: int) -> str:
        """生成单个候选 — 委托 _generate_batch（逻辑统一，无代码重复）"""
        return self._generate_batch(prompt, num_candidates=1, seed=seed)[0]

    def _generate_batch(self, prompt: str, num_candidates: int, seed: int) -> list[str]:
        """
        批量生成多个候选 — 单次 model.generate() 调用，GPU 满载。

        原理: 将 prompt 在 batch 维度复制 num_candidates 份，
        do_sample=True 时 torch.multinomial 按 batch 元素独立采样，保证多样性。
        内置 torch.no_grad() 防止生成阶段残留计算图（显存安全）。
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        messages = [
            {"role": "system", "content": "你是一个都市文娱小说作家，文风轻松诙谐、爽感十足。"},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 截断：文本层面粗略截断 + tokenizer truncation 精确截断（自动同步 attention_mask）
        enc_check = self.tokenizer(text, return_tensors="pt")
        if enc_check["input_ids"].shape[1] > self.config.max_prompt_length:
            char_limit = self.config.max_prompt_length * 3
            text = text[-char_limit:]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=self.config.max_prompt_length,
                               return_attention_mask=True)
        # 在 batch 维度复制 num_candidates 份
        input_ids = inputs["input_ids"].repeat(num_candidates, 1).to(self.device)
        attention_mask = inputs["attention_mask"].repeat(num_candidates, 1).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,                  # 随机采样保证多样性
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # 解码每个候选
        candidates = []
        prompt_len = input_ids.shape[1]
        for i in range(num_candidates):
            resp = self.tokenizer.decode(
                outputs[i][prompt_len:], skip_special_tokens=True
            )
            candidates.append(resp)
        return candidates

    def _tokenize_candidate(self, candidate: str, prompt: str):
        """Tokenize 候选文本，返回 (input_ids, attention_mask, prompt_len)"""
        messages = [
            {"role": "system", "content": "你是一个都市文娱小说作家，文风轻松诙谐、爽感十足。"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": candidate},
        ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_enc = self.tokenizer.apply_chat_template(
            messages[:2], add_generation_prompt=True, tokenize=False
        )
        prompt_len = len(self.tokenizer(prompt_enc)["input_ids"])

        enc = self.tokenizer(full_text, return_tensors="pt", truncation=True,
                            max_length=self.config.max_prompt_length + self.config.max_new_tokens)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device) if "attention_mask" in enc else None

        # 验证 prompt 边界对齐
        prompt_enc_ids = self.tokenizer(prompt_enc, return_tensors="pt")["input_ids"]
        actual_prompt_len = prompt_enc_ids.shape[1]
        if actual_prompt_len <= input_ids.shape[1]:
            if not torch.equal(input_ids[:, :actual_prompt_len], prompt_enc_ids.to(self.device)):
                actual_prompt_len = min(actual_prompt_len, input_ids.shape[1] // 2)
            prompt_len = actual_prompt_len

        return input_ids, attention_mask, prompt_len

    def _compute_grpo_loss(self, candidate: str, prompt: str, advantage: float) -> torch.Tensor:
        """
        GRPO loss: advantage × nll + kl_penalty × ||lora||²

        - 第一项: 策略梯度 — 强化高 reward、抑制低 reward 的生成模式
        - 第二项: KL 正则化 — L2 近似，防止模型偏离 SFT 太远
        """
        input_ids, attention_mask, prompt_len = self._tokenize_candidate(candidate, prompt)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # 只对 assistant 部分计算 loss

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        nll = torch.nn.functional.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            labels.view(-1), ignore_index=-100, reduction="mean",
        )

        # 推导: nll = -log P, 策略梯度最小化 loss = -advantage × log P = advantage × nll
        loss = advantage * nll

        # KL 正则化项（L2 近似：约束 LoRA 权重幅度，代理 KL 散度）
        lora_l2 = 0.0
        for n, p in self.model.named_parameters():
            if "lora" in n and p.requires_grad:
                lora_l2 += torch.sum(p ** 2)
        loss = loss + self.config.kl_penalty * lora_l2

        return loss

    def _apply_kl_penalty(self):
        """已废弃 — KL 惩罚已合并进 _compute_grpo_loss，保留空方法避免调用报错"""
        pass

    def _evaluate(self, episode: int):
        """评估当前模型"""
        self.model.eval()
        prompts = self.prompt_pool.sample(5)
        rewards = []
        for idx, p in enumerate(prompts):
            with torch.no_grad():
                text = self._generate(p, seed=episode * 100 + idx)
                r = self.reward_model.score(text)
            rewards.append(r)
        avg = sum(rewards) / len(rewards)
        # 用 tqdm.write 避免破坏进度条（_evaluate 在 tqdm 循环内调用）
        from tqdm import tqdm as _tqdm
        _tqdm.write(f"  📊 Eval@{episode}: avg_reward={avg:.3f} | "
                    f"best_so_far={self.best_avg_reward:.3f}")
        self.model.train()

    def _load_state(self):
        """从 checkpoint 恢复训练状态"""
        latest_path = Path(self.config.output_dir) / "latest" / "grpo_state.json"
        best_path = Path(self.config.output_dir) / "best" / "grpo_state.json"
        # 优先 latest（最近进度），其次 best
        state_path = latest_path if latest_path.exists() else best_path
        if not state_path.exists():
            print("ℹ️ 未找到 checkpoint，从头开始训练")
            return
        try:
            state = json.loads(state_path.read_text())
            saved_episodes = state.get("episodes_completed", 0)
            self._start_episode = saved_episodes + 1
            self.best_avg_reward = state.get("best_avg_reward", 0.0)
            self.train_rewards = state.get("train_rewards", [])
            print(f"📂 从 episode {self._start_episode} 恢复训练 "
                  f"(best_reward={self.best_avg_reward:.4f}, history={len(self.train_rewards)} 条)")
        except Exception as e:
            print(f"⚠️ 恢复失败 ({e})，从头开始训练")

    def _save(self, tag: str):
        path = Path(self.config.output_dir) / tag
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # 保存训练状态
        (path / "grpo_state.json").write_text(json.dumps({
            "episodes_completed": len(self.train_rewards),
            "best_avg_reward": self.best_avg_reward,
            "train_rewards": self.train_rewards[-100:],
        }, indent=2))
        print(f"  💾 保存到 {path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO 强化学习训练")
    parser.add_argument("--model", default="models/novel-merged")
    parser.add_argument("--lora", default=None, help="SFT LoRA 权重路径")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="每次生成长度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="梯度累积步数（自动确保>=group_size）")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl", type=float, default=0.05)
    parser.add_argument("--output_dir", default="output/grpo")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed 自动注入，不要手动设置")
    parser.add_argument("--max_concurrent", type=int, default=5, help="DeepSeek API 最大并发请求数")
    parser.add_argument("--dry_run", action="store_true", help="只测试奖励模型")
    parser.add_argument("--resume", action="store_true", help="从最近 checkpoint 恢复训练")
    args = parser.parse_args()

    config = GRPOConfig(
        model_path=args.model,
        sft_lora_path=args.lora,
        total_episodes=args.episodes,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.lr,
        kl_penalty=args.kl,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        **({"gradient_accumulation_steps": args.gradient_accumulation_steps}
           if args.gradient_accumulation_steps is not None else {}),
    )

    if args.dry_run:
        print("🧪 测试奖励模型...")
        rm = RewardModel()
        test_text = ("重生2008年，林风睁开眼，发现自己回到了北电导演系的宿舍。"
                     "手机屏幕亮着——是前女友的分手短信。他还没来得及感慨，"
                     "脑海中突然响起一道机械音：【文娱之神系统已激活】")
        score = rm.score(test_text * 3)  # 重复到足够字数
        print(f"reward={score:.3f}")
        import sys; sys.exit(0)

    trainer = GRPOTrainer(config, resume=args.resume)
    trainer.train()
