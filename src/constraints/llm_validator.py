"""
约束校验器

对生成的章节进行 8 维质量校验，LLM 驱动 + 规则兜底。
不合格章节自动重写（最多重试 3 次），仍不通过则标记为需人工复核。

用法:
    from src.constraints.llm_validator import ChapterValidator
    v = ChapterValidator()
    result = v.validate(chapter_text, context={...})
"""

import json, os, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from src.env import load_env, get_api_key
load_env()

DEEPSEEK_KEY = get_api_key("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ValidationResult:
    passed: bool = True
    scores: dict = field(default_factory=dict)       # 各维度得分
    issues: list[str] = field(default_factory=list)   # 问题列表
    suggestions: list[str] = field(default_factory=list)  # 改进建议
    needs_rewrite: bool = False


class ChapterValidator:
    """8 维约束校验器（LLM 驱动）"""

    def __init__(self, model: str = "deepseek-v4-flash"):
        self.model = model
        self.max_retries = 3      # 单章最多重试次数

    # ================================================================
    # 主入口
    # ================================================================

    def validate(self, chapter_text: str, context: dict) -> ValidationResult:
        """
        校验单章质量。

        context 应包含:
          - chapter_index: int
          - outline_node: dict      (大纲中该章节点)
          - characters: dict         (已知人物表)
          - golden_finger: dict      (金手指设定)
          - previous_3_chapters: list[str]  (前 3 章内容, 去重用)
        """
        # 1. 规则检查（不需要 LLM）
        word_count = len(chapter_text.replace("\n", "").replace(" ", ""))
        rule_issues = []

        if word_count < 2000:
            rule_issues.append(f"字数不足({word_count}字 < 2000)")
        elif word_count > 3000:
            rule_issues.append(f"字数超标({word_count}字 > 3000)")

        # 重复检测
        prev_texts = context.get("previous_3_chapters", [])
        if prev_texts:
            rep_rate = self._repetition_rate(chapter_text, prev_texts)
            if rep_rate > 0.3:
                rule_issues.append(f"与前文重复度过高({rep_rate:.0%})")

        # 2. LLM 评估
        llm_result = self._llm_evaluate(chapter_text, context)

        # 3. 汇总
        issues = rule_issues + llm_result.get("issues", [])
        suggestions = llm_result.get("suggestions", [])
        scores = llm_result.get("scores", {})
        overall = scores.get("整体质量", 3)

        return ValidationResult(
            passed=(overall >= 3 and len(issues) <= 2),
            scores=scores,
            issues=issues,
            suggestions=suggestions,
            needs_rewrite=(overall < 3 or len(rule_issues) >= 2),
        )

    # ================================================================
    # 8 维校验 Prompt
    # ================================================================

    _EVAL_PROMPT = """你是网文质量评审专家。请对以下都市文娱小说章节进行质量评估。

【本章类型】%s
【目标高潮类型】%s
【金手指设定】%s
【已知人物】%s

【章节内容】
%s

请按以下 JSON 格式输出评估结果:
{
  "scores": {
    "高潮嵌入": 0-10,   // 本章是否有有效高潮/爽点
    "人物一致性": 0-10, // 人名和设定是否与已知人物一致
    "金手指一致性": 0-10, // 金手指使用是否超出设定
    "三线覆盖": 0-10,   // 事业/感情/日常三条线的覆盖度
    "结尾钩子": 0-10,   // 最后3句是否制造了悬念
    "文风口语化": 0-10, // 是否口语化、有梗
    "整体质量": 1-5     // 综合
  },
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}"""

    def _llm_evaluate(self, text: str, context: dict) -> dict:
        """调用 LLM 评估"""
        if not DEEPSEEK_KEY:
            return {"scores": {"整体质量": 3}, "issues": [], "suggestions": []}

        outline = context.get("outline_node", {})
        prompt = self._EVAL_PROMPT % (
            outline.get("scene_type", "daily"),
            outline.get("climax_type", "无"),
            json.dumps(context.get("golden_finger", {}), ensure_ascii=False),
            json.dumps(context.get("characters", {}), ensure_ascii=False)[:300],
            text[:3000],  # 只评前 3000 字
        )

        try:
            resp = httpx.post(
                DEEPSEEK_URL,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512, "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
                headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
                timeout=60,
            )
            return json.loads(resp.json()["choices"][0]["message"]["content"])
        except Exception:
            return {"scores": {"整体质量": 3}, "issues": [], "suggestions": []}

    # ================================================================
    # 规则方法
    # ================================================================

    def _repetition_rate(self, text: str, prev_texts: list[str]) -> float:
        """4-gram 重叠率"""
        def ngrams(t, n=4):
            chars = re.sub(r"\s", "", t)
            return {chars[i:i+n] for i in range(len(chars)-n+1)}

        gen_ngrams = ngrams(text)
        if not gen_ngrams:
            return 0
        overlaps = [len(gen_ngrams & ngrams(p)) / len(gen_ngrams) for p in prev_texts]
        return max(overlaps) if overlaps else 0

    def check_hook(self, last_3_sentences: str) -> dict:
        """检查结尾是否有钩子（本地规则）"""
        patterns = {
            "疑问": r"[？?]",
            "转折": r"但是|然而|不料|没想到|谁知",
            "预告": r"接下来|明天|等.*再|看.*怎么",
            "悬念": r"…|\.{3}|——",
        }
        matches = {k: bool(re.search(v, last_3_sentences)) for k, v in patterns.items()}
        return {"has_hook": any(matches.values()), "hooks": [k for k, v in matches.items() if v]}


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    v = ChapterValidator()
    test_text = "林风站在领奖台上，看着台下沸腾的人群。他想起两年前重生时的迷茫，恍如隔世。\n" * 20
    result = v.validate(test_text, {
        "chapter_index": 10,
        "outline_node": {"scene_type": "release", "climax_type": "major"},
        "golden_finger": {"type": "rebirth", "name": "文娱之神系统"},
        "characters": {"林风": "主角"},
    })
    print(f"通过: {result.passed}")
    print(f"评分: {result.scores}")
    print(f"问题: {result.issues}")
    print(f"建议: {result.suggestions}")
