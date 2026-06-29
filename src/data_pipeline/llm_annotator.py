"""
LLM 标注模块

使用外部高质量 LLM (DeepSeek V4 Pro / GPT-5.5) 对清洗后的章节进行:
- 场景类型分类（日常/作品发布/综艺录制/商战/感情/打脸）
- 叙事节奏标注（铺垫/小高潮/大高潮/过渡/收尾）
- 实体抽取（主角名、金手指类型、作品名、CP名）
- 三线覆盖评分（事业线/感情线/日常线）
- 质量评分（1-5分）

标注策略按任务主观性分层:
- 客观任务（场景分类、实体抽取）: LLM 独立标注，一致率目标 ≥85%
- 主观任务（高潮判断、质量评分）: LLM 初筛 + 人工对边界 case 纠偏，一致率目标 ≥75%
"""

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx


# ============================================================
# .env 加载（避免额外依赖 python-dotenv）
# ============================================================

def _load_env():
    """从项目根目录加载 .env 文件"""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                os.environ.setdefault(key.strip(), value.strip())
_load_env()


# ============================================================
# 数据结构
# ============================================================

@dataclass
class AnnotationResult:
    """单章 LLM 标注结果"""
    chapter_index: int
    # 场景类型（可多标签）
    scene_types: list[str] = field(default_factory=list)
    primary_scene: str = ""

    # 叙事节奏
    rhythm_label: str = ""              # buildup | mini_climax | major_climax | transition | resolution
    rhythm_confidence: float = 0.0

    # 实体抽取
    entities: dict = field(default_factory=dict)
    golden_finger_type: str = ""        # rebirth | system | copycat | talent
    golden_finger_name: str = ""
    key_works: list[str] = field(default_factory=list)
    cp_pairs: list[dict] = field(default_factory=list)

    # 三线覆盖 (0-10)
    career_score: int = 0
    romance_score: int = 0
    daily_score: int = 0

    # 质量评分 (1-5)
    quality_score: int = 0
    quality_notes: str = ""

    # 高潮信息（如果是高潮章）
    climax_type: str = ""               # 打脸 | 逆袭 | 公布 | 获奖 | 突破
    climax_intensity: int = 0           # 1-5

    # 结尾钩子
    has_hook: bool = False
    hook_type: str = ""                 # 疑问 | 转折 | 危机 | 预告

    # 标注元信息
    annotator_model: str = ""
    annotation_cost: float = 0.0        # API 调用费用


# ============================================================
# Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一个专业的网络文学分析专家，擅长分析都市文娱类小说。
你的任务是分析给定的章节内容，并输出结构化的 JSON 标注结果。

## 标注规范

### 1. 场景类型 (scene_types)
从以下类别中选择（可多选，按相关度排序）:
- daily: 日常互动（聊天、吃饭、聚会等轻松日常）
- release: 作品发布（歌曲上线、电影上映、专辑发布）
- variety: 综艺录制（综艺节目拍摄或播出）
- business: 商战谈判（投资、合同、公司经营）
- romance: 感情戏（恋爱互动、修罗场、感情抉择）
- face_slapping: 打脸场面（证明自己、打脸质疑者、反转逆袭）

primary_scene 选最主要的那个。

### 2. 叙事节奏 (rhythm_label)
- buildup: 铺垫章（推进剧情，未到爆发点）
- mini_climax: 小高潮章（单元内阶段性成果）
- major_climax: 大高潮章（重大突破）
- transition: 过渡章（节奏放缓）
- resolution: 收尾章（单元收尾）

### 3. 金手指
判断金手指类型:
- rebirth: 重生先知
- system: 系统任务
- copycat: 文抄公（搬运地球作品）
- talent: 天赋异禀

### 4. 三线评分 (0-10)
- career_score: 事业线强度
- romance_score: 感情线强度
- daily_score: 日常线强度

### 5. 质量评分 (1-5)
- 5: 文风成熟、节奏紧凑、爽感十足、有记忆点
- 4: 整体不错，有小瑕疵
- 3: 中规中矩，无明显亮点
- 2: 质量偏低，有明显问题
- 1: 质量差，不应作为训练样本

### 6. 高潮信息
如果本章是小高潮或大高潮，标注:
- climax_type: 打脸 | 逆袭 | 公布 | 获奖 | 突破
- climax_intensity: 1-5

### 7. 结尾钩子
判断最后 3 句是否制造了悬念/期待:
- has_hook: true/false
- hook_type: 疑问 | 转折 | 危机 | 预告

## 输出格式
请严格按照以下 JSON 格式输出，不要包含 ```json 标记:"""


def build_annotation_prompt(chapter_text: str, chapter_index: int) -> str:
    """构造单章标注的 user prompt"""
    # 截断过长文本（取前 4000 字和后 500 字，保留完整结构）
    if len(chapter_text) > 5000:
        text = chapter_text[:4000] + "\n\n... [中间省略] ...\n\n" + chapter_text[-500:]
    else:
        text = chapter_text

    return f"""请分析以下小说的第 {chapter_index} 章。

【章节内容】
{text}

请输出 JSON 格式的标注结果。"""


# ============================================================
# LLM 客户端
# ============================================================

class LLMAnnotator:
    """LLM 标注器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",  # DeepSeek V4 Pro
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 费用统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _call_api(self, system_prompt: str, user_prompt: str) -> dict:
        """调用 LLM API"""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # 低温度，追求一致的标注结果
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=120) as client:
                    response = client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                    # 累计 token 用量
                    usage = data.get("usage", {})
                    self.total_input_tokens += usage.get("prompt_tokens", 0)
                    self.total_output_tokens += usage.get("completion_tokens", 0)

                    content = data["choices"][0]["message"]["content"]
                    return json.loads(content)

            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"LLM API 调用失败（已重试 {self.max_retries} 次）: {e}")

    def annotate_chapter(
        self,
        chapter_text: str,
        chapter_index: int,
    ) -> AnnotationResult:
        """标注单章"""
        user_prompt = build_annotation_prompt(chapter_text, chapter_index)

        try:
            raw = self._call_api(SYSTEM_PROMPT, user_prompt)

            result = AnnotationResult(
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

            return result

        except Exception as e:
            # 返回默认标注（后续人工复核）
            return AnnotationResult(
                chapter_index=chapter_index,
                quality_notes=f"标注失败: {e}",
                annotator_model=self.model,
            )

    def annotate_book(
        self,
        book_dir: Path,
        resume: bool = True,
    ) -> list[AnnotationResult]:
        """
        标注一整本书的所有章节

        Args:
            book_dir: 清洗后的书籍目录 (data/clean/{book_name}/)
            resume: 是否从上次中断处继续

        Returns:
            list[AnnotationResult]
        """
        # 加载已有结果
        output_file = book_dir / "llm_annotations.json"
        existing_results: dict[int, AnnotationResult] = {}
        if resume and output_file.exists():
            existing_data = json.loads(output_file.read_text('utf-8'))
            for item in existing_data:
                existing_results[item["chapter_index"]] = AnnotationResult(**item)

        # 获取章节列表
        chapter_files = sorted(book_dir.glob("chapter_*.txt"))
        results = []

        for ch_file in chapter_files:
            # 提取章节号
            ch_num = int(ch_file.stem.split('_')[-1])

            if resume and ch_num in existing_results:
                results.append(existing_results[ch_num])
                print(f"  ⏭️ 第{ch_num}章: 已有标注，跳过")
                continue

            chapter_text = ch_file.read_text('utf-8')
            print(f"  📝 第{ch_num}章: 标注中...", end=' ')

            annotation = self.annotate_chapter(chapter_text, ch_num)
            results.append(annotation)

            print(f"场景={annotation.primary_scene}, "
                  f"节奏={annotation.rhythm_label}, "
                  f"质量={annotation.quality_score}/5")

            # 每次标注后立即保存（防止中断丢失）
            output_file.write_text(
                json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

            # 速率控制
            time.sleep(0.5)

        return results

    def compute_cost(self) -> dict:
        """计算累计费用（基于 DeepSeek V4 Pro 定价）"""
        # DeepSeek V4 Pro: 输入 $0.145/M, 输出 $3.48/M
        input_cost = self.total_input_tokens / 1_000_000 * 0.145
        output_cost = self.total_output_tokens / 1_000_000 * 3.48
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
        }


# ============================================================
# 批量标注入口
# ============================================================

def annotate_all_books(
    clean_dir: Path,
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    limit_chapters_per_book: Optional[int] = None,
) -> None:
    """
    批量标注 data/clean/ 下的所有书籍

    Args:
        clean_dir: data/clean/ 目录
        api_key: DeepSeek API Key
        model: 模型名称
        limit_chapters_per_book: 每本书最多标注章数（用于小规模验证）
    """
    annotator = LLMAnnotator(api_key=api_key, model=model)

    book_dirs = sorted(d for d in clean_dir.iterdir() if d.is_dir())

    total_books = len(book_dirs)
    for i, book_dir in enumerate(book_dirs):
        print(f"\n📚 [{i+1}/{total_books}] {book_dir.name}")

        if limit_chapters_per_book:
            # 临时重命名超出范围的章节文件
            chapter_files = sorted(book_dir.glob("chapter_*.txt"))
            for ch_file in chapter_files[limit_chapters_per_book:]:
                ch_file.rename(ch_file.with_suffix('.txt.skipped'))

        try:
            annotator.annotate_book(book_dir)
        finally:
            # 恢复被跳过的章节
            for skipped in book_dir.glob("*.txt.skipped"):
                skipped.rename(skipped.with_suffix('.txt'))

    cost = annotator.compute_cost()
    print(f"\n💰 累计费用: ${cost['estimated_cost_usd']:.2f} "
          f"(input: {cost['input_tokens']:,} tokens, "
          f"output: {cost['output_tokens']:,} tokens)")


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import sys

    clean_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "data" / "clean"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None  # 每本最多 N 章，用于 Phase 1.5

    annotate_all_books(clean_dir, limit_chapters_per_book=limit)
