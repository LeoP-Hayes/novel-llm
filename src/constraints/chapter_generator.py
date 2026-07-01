"""
章节生成器

串联大纲规划器 + 模型推理 + 约束校验的完整生成流程。

工作流:
  1. 读取大纲节点 → 确定本章目标
  2. 拼接前文 + RAG 检索（可选）→ 构造 prompt
  3. 调用模型生成章节
  4. 约束校验 → 不通过则重写（最多 3 次）
  5. 更新知识库（人物/作品/前情）

用法:
    from src.constraints.chapter_generator import NovelGenerator
    gen = NovelGenerator(model_path="models/novel-merged")
    gen.generate(user_prompt="...", chapters=10)

依赖: outline_planner.py + llm_validator.py
模型: 本地路径或 AutoDL 路径
"""

import gc, json, os, re, time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .outline_planner import OutlinePlanner
from .llm_validator import ChapterValidator, ValidationResult


# ============================================================
# 配置
# ============================================================

SYSTEM_PROMPT = (
    "你是一个都市文娱小说作家，文风轻松诙谐、爽感十足。"
    "语言口语化、有梗、拒绝说教。每章 2000-3000 字，起承转合，结尾留钩子。"
    "三线并进：事业线（文娱成就）+ 感情线（恋爱/修罗场）+ 日常线（轻松互动）。"
)


class NovelGenerator:
    """都市文娱小说生成器"""

    def __init__(
        self,
        model_path: str = "models/novel-merged",
        validator_model: str = "deepseek-v4-flash",
    ):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.planner = OutlinePlanner()
        self.validator = ChapterValidator(model=validator_model)
        self.knowledge = {"characters": {}, "works": [], "summary": ""}
        self.output_dir = Path("output/novel")

    # ================================================================
    # 主入口
    # ================================================================

    def generate(
        self,
        user_prompt: str,
        chapters: int = 10,
        output_dir: Optional[Path] = None,
    ) -> list[dict]:
        """生成完整小说"""
        if output_dir:
            self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 大纲规划
        outline = self.planner.plan(user_prompt, chapters=chapters)
        outline_path = self.output_dir / "outline.json"
        outline_path.write_text(json.dumps(outline, ensure_ascii=False, indent=2))
        print(f"📋 大纲已保存: {outline_path}")

        # 2. 加载模型
        self._load_model()

        # 3. 逐章生成
        generated = []
        full_text = outline["prompt"] + "\n\n"

        for node in outline["chapters"]:
            ch_num = node["chapter"]
            print(f"\n{'='*50}")
            print(f"📝 第{ch_num}章 [{node.get('climax_type','平')}] {node.get('summary','')[:40]}")

            # 生成本章（含重试）
            chapter_text = self._generate_with_retry(
                ch_num, node, generated, outline, max_retries=3
            )

            # 保存
            ch_data = {"num": ch_num, "text": chapter_text, "outline_node": node}
            generated.append(ch_data)

            ch_path = self.output_dir / f"chapter_{ch_num:02d}.txt"
            ch_path.write_text(f"第{ch_num}章\n\n{chapter_text}", encoding="utf-8")
            full_text += f"\n\n第{ch_num}章\n{chapter_text}"
            print(f"  ✅ {len(chapter_text)} 字 → {ch_path.name}")

            # 更新知识库
            self._update_knowledge(chapter_text, ch_num)

            # 释放显存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4. 保存全文
        full_path = self.output_dir / "full.txt"
        full_path.write_text(full_text, encoding="utf-8")
        print(f"\n✅ 全文: {full_path} ({sum(len(c['text']) for c in generated)} 字)")

        return generated

    # ================================================================
    # 内部
    # ================================================================

    def _load_model(self):
        if self.model is not None:
            return
        print(f"🤖 加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=True, device_map="auto"
        ).eval()

    def _generate_with_retry(self, ch_num, node, prev, outline, max_retries):
        """生成本章 + 校验 + 重试"""
        for attempt in range(max_retries + 1):
            # 构造 prompt
            prompt = self._build_chapter_prompt(ch_num, node, prev, outline)

            # 生成
            chapter_text = self._call_model(prompt)

            # 校验
            context = {
                "chapter_index": ch_num,
                "outline_node": node,
                "characters": self.knowledge.get("characters", {}),
                "golden_finger": self._extract_golden_finger(outline),
                "previous_3_chapters": [c["text"][-800:] for c in prev[-3:]],
            }
            result = self.validator.validate(chapter_text, context)

            if result.passed or attempt == max_retries:
                if not result.passed:
                    print(f"  ⚠️ 校验不通过但已达重试上限: {result.issues}")
                return chapter_text

            print(f"  🔄 重试 {attempt+1}/{max_retries}: {result.issues[:2]}")

    def _build_chapter_prompt(self, ch_num, node, prev, outline) -> list[dict]:
        """构造 ChatML prompt"""
        # 前情上下文
        context = ""
        if prev:
            recent = prev[-3:]
            for c in recent:
                snippet = c["text"][-500:] if len(c["text"]) > 500 else c["text"]
                context += f"\n【第{c['num']}章最后片段】\n...{snippet}\n"

        # 三线指令
        lines = node.get("lines", {})
        line_desc = ""
        if lines:
            line_desc = f"事业线: {lines.get('career','推进')} | 感情线: {lines.get('romance','保持')} | 日常线: {lines.get('daily','轻松')}"

        # 高潮指令
        climax_hint = {
            "mini": "本章需要一个小高潮——阶段性成果、反转或打脸。",
            "major": "⚠️ 本章是大高潮章——重大突破、获奖或行业地位跃升！需要爆点。",
        }.get(node.get("climax_type", ""), "")

        user_msg = (
            f"【小说设定】{outline['prompt'][:500]}\n"
            f"{context}\n"
            f"【第{ch_num}章大纲】{node.get('summary', '自由发挥')}\n"
            f"{line_desc}\n"
            f"{climax_hint}\n"
            f"【写作要求】2000-3000字，起承转合，结尾留钩子。口语化、有梗。"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _call_model(self, messages: list[dict]) -> str:
        """调用模型生成"""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=2000, temperature=0.8, top_p=0.9,
                do_sample=True, repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _update_knowledge(self, text: str, ch_num: int):
        """增量更新知识库（简易版）"""
        # 提取新人名（中文字符 2-4 字 + 不是常见词）
        names = set(re.findall(r"[林刘陈王李张赵周吴郑杨许何吕施][一-鿿]{1,2}", text))
        for name in names:
            if name not in self.knowledge["characters"]:
                self.knowledge["characters"][name] = f"第{ch_num}章出场"
        # 更新摘要
        self.knowledge["summary"] = f"已写到第{ch_num}章。" + text[:200]

    def _extract_golden_finger(self, outline: dict) -> dict:
        """从大纲提取金手指信息"""
        prompt = outline.get("prompt", "")
        if "系统" in prompt:
            return {"type": "system", "name": "文娱之神系统", "desc": "完成任务获得奖励"}
        if "重生" in prompt:
            return {"type": "rebirth", "era": "2008"}
        return {"type": "unknown"}


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="重生2008年，北电导演系学生林风绑定文娱系统，用前世记忆征服娱乐圈。")
    p.add_argument("--chapters", type=int, default=10)
    p.add_argument("--model", default="models/novel-merged")
    p.add_argument("--output", default="output/novel")
    args = p.parse_args()

    gen = NovelGenerator(model_path=args.model)
    gen.generate(args.prompt, chapters=args.chapters, output_dir=Path(args.output))
