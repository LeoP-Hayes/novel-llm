"""
SFT 训练数据构造
从清洗后的章节 + LLM 标注 → 5 种任务类型的 ChatML 训练样本
输出: data/sft/train.jsonl + val.jsonl
"""

import json, random, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict


@dataclass
class BuildConfig:
    total_target: int = 6000
    min_quality: int = 3
    val_ratio: float = 0.1
    max_context_tokens: int = 5000
    chapter_min_words: int = 800
    chapter_max_words: int = 5000
    continuation_n: int = 2
    task_ratios: dict = field(default_factory=lambda: {
        "continuation": 0.30, "unit_creation": 0.25, "scene_generation": 0.20,
        "style_imitation": 0.15, "outline_expansion": 0.10,
    })
    random_seed: int = 42


SYSTEM_PROMPTS = {
    "continuation": "你是一个都市文娱小说作家，擅长轻松诙谐的文风。请根据前文续写下一章。2000-3000字，起承转合，结尾留钩子，三线并进。",
    "unit_creation": "你是一个都市文娱小说作家。请根据给定的大纲完成一个完整的叙事单元，包含铺垫、小高潮、收尾，节奏紧凑。",
    "scene_generation": "你是一个都市文娱小说作家。请根据场景设定和情境生成对应的场景内容。语言口语化、有梗、有爽感。",
    "style_imitation": "你是一个都市文娱小说作家。请模仿以下段落的文风和节奏写一段新的都市文娱场景。",
    "outline_expansion": "你是一个都市文娱小说作家。请根据给定的大纲扩展为完整的章节内容。",
}


def load_book_data(book_dir: Path) -> dict:
    chapters = {}
    annotations = {}
    ann_file = book_dir / "llm_annotations.json"
    if ann_file.exists():
        for a in json.loads(ann_file.read_text('utf-8')):
            annotations[a["chapter_index"]] = a

    for ch_file in sorted(book_dir.glob("chapter_*.txt")):
        name = ch_file.stem
        if name.endswith('.txt'):
            continue
        try:
            ch_num = int(name.split('_')[-1])
        except ValueError:
            continue
        text = ch_file.read_text('utf-8')
        chapters[ch_num] = {
            "text": text,
            "word_count": len(text.replace('\n', '').replace(' ', '')),
            "annotation": annotations.get(ch_num),
        }

    meta_file = book_dir / "metadata.json"
    meta = json.loads(meta_file.read_text('utf-8')) if meta_file.exists() else {}
    return {"book_name": book_dir.name, "author": meta.get("author", "unknown"), "chapters": chapters}


def load_all_books(clean_dir: Path, config: BuildConfig) -> list[dict]:
    books = []
    for book_dir in sorted(clean_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        data = load_book_data(book_dir)
        valid = {}
        for k, v in data["chapters"].items():
            if not (config.chapter_min_words <= v["word_count"] <= config.chapter_max_words):
                continue
            ann = v.get("annotation")
            if ann and ann.get("quality_score", 0) < config.min_quality:
                continue
            valid[k] = v
        if len(valid) >= 10:
            data["chapters"] = valid
            books.append(data)
    return books


def build_continuation_sample(book: dict, ch_num: int, config: BuildConfig) -> Optional[dict]:
    chapters = book["chapters"]
    sorted_keys = sorted(chapters.keys())
    try:
        target_idx = sorted_keys.index(ch_num)
    except ValueError:
        return None
    start_idx = max(0, target_idx - config.continuation_n + 1)
    prev_ch_nums = sorted_keys[start_idx:target_idx]
    if len(prev_ch_nums) < config.continuation_n - 1:
        return None

    context_parts = []
    for ch in [chapters[i] for i in prev_ch_nums]:
        text = ch["text"]
        snippet = "..." + text[-1000:] if len(text) > 1000 else text
        context_parts.append(f"【前文片段】\n{snippet}")
    context = "\n\n".join(context_parts)
    if len(context) > config.max_context_tokens:
        context = context[-config.max_context_tokens:]

    instr = f"请根据以上前文续写第{ch_num}章。要求2000-3000字，有起承转合，结尾留悬念。"
    return {"task_type": "continuation", "system": SYSTEM_PROMPTS["continuation"],
            "user": f"{context}\n\n{instr}", "assistant": chapters[ch_num]["text"]}


def build_scene_generation_sample(book: dict, ch_num: int, config: BuildConfig) -> Optional[dict]:
    chapters = book["chapters"]
    ch = chapters.get(ch_num)
    if not ch or not ch.get("annotation"):
        return None
    ann = ch["annotation"]
    scene_names = {"daily":"日常互动","release":"作品发布","variety":"综艺录制",
                   "business":"商战谈判","romance":"感情戏","face_slapping":"打脸场面"}
    scene_cn = scene_names.get(ann.get("primary_scene", "daily"), "日常互动")
    text = ch["text"]
    context = text[:200] if len(text) > 200 else text
    instr = f"【场景类型】{scene_cn}\n【当前情境】{context}...\n请根据以上设定展开写一段{scene_cn}场景。"
    return {"task_type": "scene_generation", "system": SYSTEM_PROMPTS["scene_generation"],
            "user": instr, "assistant": text}


def build_outline_expansion_sample(book: dict, ch_num: int, config: BuildConfig) -> Optional[dict]:
    chapters = book["chapters"]
    ch = chapters.get(ch_num)
    if not ch or not ch.get("annotation"):
        return None
    ann = ch["annotation"]
    rhythm_desc = {"buildup":"铺垫推进剧情","mini_climax":"小高潮（阶段成果/反转）",
                   "major_climax":"大高潮（重大突破）","transition":"过渡舒缓节奏",
                   "resolution":"收尾过渡"}.get(ann.get("rhythm_label","buildup"),"推进剧情")
    outline = f"章节序号: 第{ch_num}章\n本章节奏: {rhythm_desc}\n主要场景: {ann.get('primary_scene','daily')}\n"
    if ann.get("climax_type"):
        outline += f"高潮类型: {ann['climax_type']}\n"
    instr = f"【本章大纲】\n{outline}\n请根据以上大纲展开写完整的章节内容（2000-3000字）。"
    return {"task_type": "outline_expansion", "system": SYSTEM_PROMPTS["outline_expansion"],
            "user": instr, "assistant": ch["text"]}


def build_unit_creation_sample(book: dict, start_ch: int, end_ch: int,
                                config: BuildConfig) -> Optional[dict]:
    chapters = book["chapters"]
    unit_chapters = {i: chapters[i] for i in range(start_ch, end_ch+1) if i in chapters}
    if len(unit_chapters) < 2:
        return None

    outline_parts, unit_text_parts = [], []
    for num, ch in sorted(unit_chapters.items()):
        text = ch["text"]
        ann = ch.get("annotation") or {}
        marker = {"mini_climax":"【小高潮】","major_climax":"【大高潮】"}.get(
            ann.get("rhythm_label",""), "")
        preview = text[:100].replace('\n', ' ')
        outline_parts.append(f"第{num}章: {marker} {ann.get('primary_scene','daily')} - {preview}...")
        unit_text_parts.append(f"第{num}章\n{text}")

    outline = "【单元大纲】\n" + "\n".join(outline_parts)
    unit_text = "\n\n".join(unit_text_parts)
    if len(unit_text) > 15000:
        return None
    instr = f"{outline}\n\n请根据以上大纲写出该单元的完整内容。"
    return {"task_type": "unit_creation", "system": SYSTEM_PROMPTS["unit_creation"],
            "user": instr, "assistant": unit_text}


def build_style_imitation_sample(books: list[dict], config: BuildConfig) -> Optional[dict]:
    book = random.choice(books)
    chapters = list(book["chapters"].values())
    if len(chapters) < 4:
        return None
    selected = random.sample(chapters, 4)
    examples = []
    for i, ch in enumerate(selected[:3]):
        text = ch["text"]
        start = random.randint(0, max(0, len(text)-400))
        examples.append(f"【示例{i+1}】\n{text[start:start+400]}")
    instr = f"作者：{book['author']}\n\n" + "\n\n".join(examples) + \
            f"\n\n请模仿以上{book['author']}的文风和节奏写一段新的都市文娱场景（2000-3000字）。"
    if len(instr) > config.max_context_tokens:
        instr = instr[:config.max_context_tokens]
    return {"task_type": "style_imitation", "system": SYSTEM_PROMPTS["style_imitation"],
            "user": instr, "assistant": selected[3]["text"]}


def build_all_samples(books: list[dict], config: Optional[BuildConfig] = None) -> list[dict]:
    if config is None:
        config = BuildConfig()
    random.seed(config.random_seed)
    samples = []

    for book in books:
        ch_nums = sorted(book["chapters"].keys())
        for ch_num in ch_nums:
            for builder in [build_continuation_sample, build_scene_generation_sample,
                           build_outline_expansion_sample]:
                s = builder(book, ch_num, config)
                if s:
                    samples.append(s)
        for i in range(0, len(ch_nums)-4, 2):
            s = build_unit_creation_sample(book, ch_nums[i], ch_nums[i+4], config)
            if s:
                samples.append(s)

    num_style = len([s for s in samples if s["task_type"] == "continuation"])
    for _ in range(max(50, num_style // 3)):
        s = build_style_imitation_sample(books, config)
        if s:
            samples.append(s)

    seen = set()
    return [s for s in samples if not (s["task_type"] + s["assistant"][:200] in seen
            or seen.add(s["task_type"] + s["assistant"][:200]))]


def format_chatml(sample: dict) -> str:
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": sample["user"]},
        {"role": "assistant", "content": sample["assistant"]},
    ]
    return json.dumps({"messages": messages, "task_type": sample["task_type"]}, ensure_ascii=False)


def balance_and_split(samples: list[dict], config: BuildConfig) -> tuple[list[dict], list[dict]]:
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["task_type"]].append(s)

    balanced = []
    for task, ratio in config.task_ratios.items():
        pool = by_type.get(task, [])
        target = int(config.total_target * ratio)
        sampled = random.sample(pool, target) if len(pool) > target else pool
        balanced.extend(sampled)
        print(f"  {task}: {len(sampled)}/{len(pool)}")

    if len(balanced) < config.total_target and by_type:
        max_task = max(by_type, key=lambda t: len(by_type[t]))
        extra = random.sample(by_type[max_task],
                              min(config.total_target - len(balanced), len(by_type[max_task])))
        balanced.extend(extra)
        print(f"  补充 {max_task}: +{len(extra)}")

    random.shuffle(balanced)
    val_size = max(50, int(len(balanced) * config.val_ratio))
    return balanced[val_size:], balanced[:val_size]


def build_sft_dataset(clean_dir: Path, output_dir: Optional[Path] = None,
                      config: Optional[BuildConfig] = None) -> tuple[Path, Path]:
    if config is None:
        config = BuildConfig()
    if output_dir is None:
        output_dir = Path.cwd() / "data" / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📂 加载数据...")
    books = load_all_books(clean_dir, config)
    print(f"  {len(books)} 本书, {sum(len(b['chapters']) for b in books)} 章有效")

    print("🔨 构造样本...")
    samples = build_all_samples(books, config)
    print(f"  共构造 {len(samples)} 条")

    print("⚖️ 均衡采样...")
    train, val = balance_and_split(samples, config)
    print(f"  train={len(train)}, val={len(val)}")

    train_path, val_path = output_dir / "train.jsonl", output_dir / "val.jsonl"
    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, 'w', encoding='utf-8') as f:
            for s in data:
                f.write(format_chatml(s) + '\n')

    task_counts = defaultdict(int)
    for s in train:
        task_counts[s["task_type"]] += 1
    print(f"\n✅ SFT 数据集: train={train_path}({len(train)}条) val={val_path}({len(val)}条)")
    print(f"   任务分布: {dict(task_counts)}")
    return train_path, val_path


if __name__ == "__main__":
    import sys
    cd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "data" / "clean"
    od = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "data" / "sft"
    build_sft_dataset(cd, od)
