"""
标注抽检工具

从全量 LLM 标注中随机抽取 300 章，生成人工复核文件。
复核完成后，计算标注一致率（客观任务 / 主观任务分别统计）。
"""

import json
import random
from pathlib import Path
from collections import defaultdict


def sample_annotations(
    clean_dir: Path = Path("data/clean"),
    n_samples: int = 300,
    seed: int = 42,
) -> list[dict]:
    """随机抽取 N 章用于人工复核"""
    random.seed(seed)

    all_chapters = []
    for book_dir in sorted(clean_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        ann_file = book_dir / "llm_annotations.json"
        if not ann_file.exists():
            continue
        annotations = json.loads(ann_file.read_text("utf-8"))

        for ann in annotations:
            ch_num = ann["chapter_index"]
            ch_file = book_dir / f"chapter_{ch_num:03d}.txt"
            if ch_file.exists():
                text = ch_file.read_text("utf-8")
                all_chapters.append({
                    "book": book_dir.name,
                    "chapter_index": ch_num,
                    "llm_annotation": ann,
                    "chapter_preview": text[:500],   # 前 500 字供快速判断
                    "chapter_full": text,            # 完整正文
                })

    # 分层抽样：主观任务 oversample
    subjective_labels = {"rhythm_label", "quality_score", "climax_type"}
    subjective = [c for c in all_chapters
                  if c["llm_annotation"].get("rhythm_label") in ("mini_climax", "major_climax")]
    objective = [c for c in all_chapters if c not in subjective]

    n_subj = min(len(subjective), n_samples * 2 // 3)  # 2/3 主观任务
    n_obj = n_samples - n_subj

    sampled = random.sample(subjective, n_subj) + random.sample(objective, n_obj)
    random.shuffle(sampled)

    return sampled


def generate_review_file(sampled: list[dict], output_path: Path):
    """生成人工复核 JSON 文件"""
    reviews = []
    for i, item in enumerate(sampled):
        ann = item["llm_annotation"]
        review = {
            "review_id": i + 1,
            "book": item["book"],
            "chapter_index": item["chapter_index"],
            "chapter_preview": item["chapter_preview"][:300],

            # LLM 标注结果
            "llm_scene_type": ann.get("primary_scene", ""),
            "llm_scene_types": ann.get("scene_types", []),
            "llm_rhythm": ann.get("rhythm_label", ""),
            "llm_quality": ann.get("quality_score", 0),
            "llm_career": ann.get("career_score", 0),
            "llm_romance": ann.get("romance_score", 0),
            "llm_daily": ann.get("daily_score", 0),
            "llm_climax_type": ann.get("climax_type", ""),
            "llm_has_hook": ann.get("has_hook", False),

            # 人工判断（待填写）
            "human_scene_type": "",        # 选填: daily/release/variety/business/romance/face_slapping
            "human_rhythm": "",             # 选填: buildup/mini_climax/major_climax/transition/resolution
            "human_quality": 0,             # 填 1-5
            "human_career": 0,              # 填 0-10
            "human_romance": 0,             # 填 0-10
            "human_daily": 0,               # 填 0-10
            "human_climax_type": "",         # 如有高潮，填类型
            "human_has_hook": False,         # true/false
            "human_notes": "",              # 备注（标注明显错误的原因等）
        }
        reviews.append(review)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    print(f"✅ 抽检文件已生成: {output_path}")
    print(f"   共 {len(reviews)} 条待复核")
    print(f"\n📋 填写说明:")
    print(f"   1. 读 chapter_preview（前 300 字），必要时参考原始章节正文")
    print(f"   2. 在 human_* 字段中填入你的判断")
    print(f"   3. 如果同意 LLM 标注，跳过该维度即可")
    print(f"   4. 只在和 LLM 标注不一致时才填写 human_* 字段")
    print(f"\n   场景类型: daily/release/variety/business/romance/face_slapping")
    print(f"   节奏标注: buildup/mini_climax/major_climax/transition/resolution")
    print(f"   质量评分: 1-5")
    print(f"   三线评分: 0-10")
    print(f"\n💡 建议: 不必一次做完，每次复核 50 条（约 30 分钟）")


def calculate_agreement(review_path: Path):
    """计算标注一致率"""
    with open(review_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    # 客观任务：场景类型
    obj_total = 0
    obj_agree = 0
    # 主观任务：节奏、质量、高潮类型
    subj_total = 0
    subj_agree = 0

    for r in reviews:
        # 场景类型（客观）
        if r["human_scene_type"]:
            obj_total += 1
            if r["human_scene_type"] == r["llm_scene_type"]:
                obj_agree += 1

        # 节奏（主观）
        if r["human_rhythm"]:
            subj_total += 1
            if r["human_rhythm"] == r["llm_rhythm"]:
                subj_agree += 1

        # 质量（主观）——允许 ±1 容差
        if r["human_quality"] > 0:
            subj_total += 1
            if abs(r["human_quality"] - r["llm_quality"]) <= 1:
                subj_agree += 1

        # 高潮类型（主观）
        if r["human_climax_type"]:
            subj_total += 1
            if r["human_climax_type"] == r["llm_climax_type"]:
                subj_agree += 1

    obj_rate = obj_agree / obj_total * 100 if obj_total > 0 else 0
    subj_rate = subj_agree / subj_total * 100 if subj_total > 0 else 0

    print(f"\n📊 标注一致率:")
    print(f"   客观任务（场景类型）: {obj_agree}/{obj_total} = {obj_rate:.1f}% (目标 ≥85%)")
    print(f"   主观任务（节奏+质量+高潮）: {subj_agree}/{subj_total} = {subj_rate:.1f}% (目标 ≥75%)")

    # 逐维度详情
    for dim, human_key, llm_key in [
        ("场景类型", "human_scene_type", "llm_scene_type"),
        ("节奏标注", "human_rhythm", "llm_rhythm"),
        ("高潮类型", "human_climax_type", "llm_climax_type"),
    ]:
        dim_total = sum(1 for r in reviews if r[human_key])
        dim_agree = sum(1 for r in reviews if r[human_key] and r[human_key] == r[llm_key])
        rate = dim_agree / dim_total * 100 if dim_total > 0 else 0
        print(f"   {dim}: {dim_agree}/{dim_total} = {rate:.1f}%")

    # 质量偏差分布
    quality_diffs = []
    for r in reviews:
        if r["human_quality"] > 0:
            quality_diffs.append(r["human_quality"] - r["llm_quality"])
    if quality_diffs:
        from collections import Counter
        diff_dist = Counter(quality_diffs)
        print(f"\n📈 质量评分偏差分布 (人工 - LLM):")
        for d in sorted(diff_dist):
            bar = "█" * diff_dist[d]
            print(f"   {d:+d}: {bar} ({diff_dist[d]}条)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] == "sample":
        # 生成抽检文件
        sampled = sample_annotations()
        generate_review_file(sampled, Path("data/sft/human_review.json"))

    elif sys.argv[1] == "score":
        # 计算一致率
        review_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/sft/human_review.json")
        calculate_agreement(review_path)

    else:
        print("用法:")
        print("  python human_review.py           # 生成抽检文件")
        print("  python human_review.py score     # 计算一致率")
