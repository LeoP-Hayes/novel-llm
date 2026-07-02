"""
LLM 标注模块 — DeepSeek API 驱动
对清洗后的章节做 5 维自动标注:
  场景类型 / 叙事节奏 / 实体抽取 / 三线覆盖评分 / 质量评分
"""

import json, os, re, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import httpx


def _load_env():
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                os.environ.setdefault(k.strip(), v.strip())
_load_env()


@dataclass
class AnnotationResult:
    chapter_index: int
    scene_types: list[str] = field(default_factory=list)
    primary_scene: str = ""
    rhythm_label: str = ""       # buildup/mini_climax/major_climax/transition/resolution
    rhythm_confidence: float = 0.0
    entities: dict = field(default_factory=dict)
    golden_finger_type: str = ""
    golden_finger_name: str = ""
    key_works: list[str] = field(default_factory=list)
    cp_pairs: list[dict] = field(default_factory=list)
    career_score: int = 0
    romance_score: int = 0
    daily_score: int = 0
    quality_score: int = 0
    quality_notes: str = ""
    climax_type: str = ""
    climax_intensity: int = 0
    has_hook: bool = False
    hook_type: str = ""
    annotator_model: str = ""
    annotation_cost: float = 0.0


SYSTEM_PROMPT = """你是一个专业的网络文学分析专家。分析给定章节，输出结构化 JSON。

标注规范:
1. scene_types: 从 [daily, release, variety, business, romance, face_slapping] 中选择，primary_scene 选最主要
2. rhythm_label: buildup/mini_climax/major_climax/transition/resolution
3. golden_finger_type: rebirth/system/copycat/talent
4. 三线评分(0-10): career_score/romance_score/daily_score
5. quality_score(1-5)
6. climax_type: 打脸/逆袭/公布/获奖/突破, climax_intensity(1-5)
7. has_hook + hook_type: 疑问/转折/危机/预告

严格按以下 JSON 格式输出:"""


def build_annotation_prompt(chapter_text: str, chapter_index: int) -> str:
    text = chapter_text[:4000] + "\n...[省略]...\n" + chapter_text[-500:] if len(chapter_text) > 5000 else chapter_text
    return f"分析第 {chapter_index} 章:\n\n{text}\n\n输出 JSON 标注结果。"


class LLMAnnotator:

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-v4-flash",
                 max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _call_api(self, system_prompt: str, user_prompt: str) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1, "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries):
            try:
                resp = httpx.post(url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                return json.loads(data["choices"][0]["message"]["content"])
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"API 调用失败: {e}")

    def annotate_chapter(self, chapter_text: str, chapter_index: int) -> AnnotationResult:
        user_prompt = build_annotation_prompt(chapter_text, chapter_index)
        try:
            raw = self._call_api(SYSTEM_PROMPT, user_prompt)
            return AnnotationResult(
                chapter_index=chapter_index,
                scene_types=raw.get("scene_types", []),
                primary_scene=raw.get("primary_scene", ""),
                rhythm_label=raw.get("rhythm_label", "buildup"),
                rhythm_confidence=raw.get("rhythm_confidence", 0.5),
                entities=raw.get("entities", {}),
                golden_finger_type=raw.get("golden_finger_type", ""),
                golden_finger_name=raw.get("golden_finger_name", ""),
                key_works=raw.get("key_works", []),
                cp_pairs=raw.get("cp_pairs", []),
                career_score=raw.get("career_score", 0),
                romance_score=raw.get("romance_score", 0),
                daily_score=raw.get("daily_score", 0),
                quality_score=raw.get("quality_score", 0),
                quality_notes=raw.get("quality_notes", ""),
                climax_type=raw.get("climax_type", ""),
                climax_intensity=raw.get("climax_intensity", 0),
                has_hook=raw.get("has_hook", False),
                hook_type=raw.get("hook_type", ""),
                annotator_model=self.model,
            )
        except Exception as e:
            return AnnotationResult(
                chapter_index=chapter_index,
                quality_notes=f"标注失败: {e}",
                annotator_model=self.model,
            )

    def annotate_book(self, book_dir: Path, resume: bool = True) -> list[AnnotationResult]:
        output_file = book_dir / "llm_annotations.json"
        existing = {}
        if resume and output_file.exists():
            for item in json.loads(output_file.read_text('utf-8')):
                existing[item["chapter_index"]] = AnnotationResult(**item)

        results = []
        for ch_file in sorted(book_dir.glob("chapter_*.txt")):
            ch_num = int(ch_file.stem.split('_')[-1])
            if resume and ch_num in existing:
                results.append(existing[ch_num])
                continue

            chapter_text = ch_file.read_text('utf-8')
            annotation = self.annotate_chapter(chapter_text, ch_num)
            results.append(annotation)

            output_file.write_text(
                json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
                encoding='utf-8')
            time.sleep(0.5)

        return results

    def compute_cost(self) -> dict:
        input_cost = self.total_input_tokens / 1_000_000 * 0.145
        output_cost = self.total_output_tokens / 1_000_000 * 3.48
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
        }


def annotate_all_books(clean_dir: Path, api_key: Optional[str] = None,
                       model: str = "deepseek-v4-flash",
                       limit_chapters_per_book: Optional[int] = None):
    annotator = LLMAnnotator(api_key=api_key, model=model)
    book_dirs = sorted(d for d in clean_dir.iterdir() if d.is_dir())
    for i, book_dir in enumerate(book_dirs):
        print(f"\n📚 [{i+1}/{len(book_dirs)}] {book_dir.name}")
        if limit_chapters_per_book:
            chapter_files = sorted(book_dir.glob("chapter_*.txt"))
            for ch_file in chapter_files[limit_chapters_per_book:]:
                ch_file.rename(ch_file.with_suffix('.txt.skipped'))
        try:
            annotator.annotate_book(book_dir)
        finally:
            for skipped in book_dir.glob("*.txt.skipped"):
                skipped.rename(skipped.with_suffix('.txt'))

    cost = annotator.compute_cost()
    print(f"\n💰 累计费用: ${cost['estimated_cost_usd']:.2f} (in:{cost['input_tokens']:,} out:{cost['output_tokens']:,})")


if __name__ == "__main__":
    import sys
    clean_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "data" / "clean"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    annotate_all_books(clean_dir, limit_chapters_per_book=limit)
