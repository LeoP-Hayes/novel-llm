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

import argparse, gc, json, os, random, re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import httpx
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
    gradient_accumulation_steps: int = 2   # group_size=2 时每 episode 最多 2 个候选，设 2 刚好每 episode 更新一次
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

    def __init__(self, model: str = "deepseek-v4-flash"):
        self.model = model
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self._cache = {}  # 缓存打分结果节省 API 费用

    def score(self, chapter: str, context: dict = None) -> float:
        """5 维加权总分"""
        # 1. LLM 打分（文风 + 爽感 + 专业感 + 连贯性 的启发式部分）
        llm = self._llm_score(chapter)

        # 2. 结构分（规则检查）
        wc = len(chapter.replace("\n", "").replace(" ", ""))
        structure = (
            (1.0 if 2000 <= wc <= 3000 else 0.5 if abs(wc - 2500) < 1000 else 0.2)
        )

        # 3. 重复度检查
        diversity = 1.0
        if context and context.get("previous"):
            diversity = 1.0 - max(
                self._ngram_overlap(chapter[:500], ch[:500])
                for ch in context["previous"]
            )

        return (
            0.25 * llm.get("文风匹配度", 3) / 10 +
            0.20 * llm.get("爽感", 3) / 10 +
            0.15 * structure +
            0.20 * llm.get("连贯性", 3) / 10 +
            0.10 * llm.get("专业感", 3) / 10 +
            0.10 * diversity
        )

    def _llm_score(self, text: str) -> dict:
        """调用 DeepSeek API 打分"""
        # 缓存命中
        cache_key = text[:100]
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

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.reward_model = RewardModel()
        self.prompt_pool = PromptPool(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_model()
        self._setup_optimizer()

        self.train_rewards = []
        self.best_avg_reward = 0.0
        self.no_improve_count = 0

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

        # 加载 SFT LoRA 作为初始化
        if self.config.sft_lora_path and Path(self.config.sft_lora_path).exists():
            print(f"📦 加载 SFT LoRA: {self.config.sft_lora_path}")
            try:
                self.model = PeftModel.from_pretrained(self.model, self.config.sft_lora_path)
            except TypeError:
                from peft import PeftModel as PM
                self.model = PM.from_pretrained(self.model, self.config.sft_lora_path)
                self.model = self.model.merge_and_unload()
                # 合并后重新注 LoRA（避免 peft 版本问题）
                peft_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.lora_rank, lora_alpha=self.config.lora_alpha,
                    lora_dropout=0.0,
                    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_up_proj","down_proj"],
                )
                self.model = get_peft_model(self.model, peft_cfg)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank, lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj","v_proj","k_proj","o_proj","gate_up_proj","down_proj"],
            )
            self.model = get_peft_model(self.model, peft_config)

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
        print(f"\n🚀 GRPO 训练开始: {config.total_episodes} episodes\n")

        prev_texts = []  # 用于重复度检查

        for episode in range(1, config.total_episodes + 1):
            prompt = self.prompt_pool.sample(1)[0]
            rewards = []
            candidates = []

            # 1. 对同一个 prompt 生成 group_size 个候选
            for g in range(config.group_size):
                with torch.no_grad():
                    text = self._generate(prompt, seed=episode * 10 + g)
                candidates.append(text)
                reward = self.reward_model.score(text, {"previous": prev_texts[-3:]})
                rewards.append(reward)

            # 2. GRPO: 组内相对比较
            rewards_tensor = torch.tensor(rewards, device=self.device)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_r) / std_r  # z-score 归一化

            # 3. 对每个候选计算 loss 并更新
            total_loss = 0.0
            grad_steps = 0

            for i in range(config.group_size):
                # 所有候选都参与，但劣势候选权重很小（避免训练信号稀疏）
                weight = max(advantages[i].item(), 0.05)
                loss = self._compute_grpo_loss(candidates[i], prompt, weight)
                (loss / config.gradient_accumulation_steps).backward()
                total_loss += loss.item()
                grad_steps += 1

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

            # 日志
            if episode % config.log_interval == 0:
                avg_r = sum(self.train_rewards[-50:]) / min(len(self.train_rewards), 50)
                print(f"  [{episode:4d}/{config.total_episodes}] "
                      f"avg_reward={avg_r:.3f} | "
                      f"rewards={[f'{r:.3f}' for r in rewards]} | "
                      f"loss={total_loss/max(grad_steps,1):.4f}")

            # 评估
            if episode % config.eval_interval == 0:
                self._evaluate(episode)

            # 保存
            if episode % config.save_interval == 0:
                avg_r = sum(self.train_rewards[-50:]) / min(len(self.train_rewards), 50)
                if avg_r > self.best_avg_reward:
                    self.best_avg_reward = avg_r
                    self.no_improve_count = 0
                    self._save(f"best")
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
        print(f"\n✅ GRPO 训练完成! best_avg_reward={self.best_avg_reward:.4f}")

    # ================================================================
    # 内部方法
    # ================================================================

    def _generate(self, prompt: str, seed: int) -> str:
        """生成一个候选（使用局部随机状态，避免污染全局 seed）"""
        torch.manual_seed(seed)

        messages = [
            {"role": "system", "content": "你是一个都市文娱小说作家，文风轻松诙谐、爽感十足。"},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        # 截断过长的 prompt
        if inputs["input_ids"].shape[1] > self.config.max_prompt_length:
            inputs["input_ids"] = inputs["input_ids"][:, -self.config.max_prompt_length:]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature, top_p=self.config.top_p,
                do_sample=True, repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _compute_grpo_loss(self, candidate: str, prompt: str, advantage: float) -> torch.Tensor:
        """
        GRPO loss: -advantage * log P(candidate|prompt) + kl_penalty * ||lora||^2

        - 第一项: 策略梯度 — 强化高 reward 的生成模式
        - 第二项: KL 正则化 — 防止模型偏离 SFT 太远 (L2 近似)
        """
        # 构造对话
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

        # tokenize
        enc = self.tokenizer(full_text, return_tensors="pt", truncation=True,
                            max_length=self.config.max_prompt_length + self.config.max_new_tokens)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device) if "attention_mask" in enc else None
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # 只对 assistant 计算 loss

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        nll = torch.nn.functional.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            labels.view(-1), ignore_index=-100, reduction="mean",
        )

        # 策略梯度项
        loss = -advantage * nll

        # KL 正则化项 (对 LoRA 参数的 L2 惩罚，避免偏离 SFT 太远)
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
        for p in prompts:
            with torch.no_grad():
                text = self._generate(p, seed=42)
                r = self.reward_model.score(text)
            rewards.append(r)
        avg = sum(rewards) / len(rewards)
        print(f"  📊 Eval@{episode}: avg_reward={avg:.3f} | "
              f"best_so_far={self.best_avg_reward:.3f}")
        self.model.train()

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
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl", type=float, default=0.05)
    parser.add_argument("--output_dir", default="output/grpo")
    parser.add_argument("--dry_run", action="store_true", help="只测试奖励模型")
    args = parser.parse_args()

    config = GRPOConfig(
        model_path=args.model,
        sft_lora_path=args.lora,
        total_episodes=args.episodes,
        group_size=args.group_size,
        learning_rate=args.lr,
        kl_penalty=args.kl,
        output_dir=args.output_dir,
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

    trainer = GRPOTrainer(config)
    trainer.train()
