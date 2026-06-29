"""
SFT 训练数据构造模块

从清洗后的章节 + LLM 标注结果，按 5 种任务类型构造训练样本:

  1. 章节续写 (30%): 前 N-1 章拼接为 prompt，第 N 章为 target
  2. 单元创作 (25%): LLM 标注的节奏边界 → 单元大纲为 prompt，单元全文为 target
  3. 场景生成 (20%): 场景标签 + 前 200 字 → 对应场景全文
  4. 风格模仿 (15%): 同作者 3 段短文 few-shot → 另一段为 target
  5. 大纲扩展 (10%): LLM 反向总结大纲 → 对应章节内容

输出: data/sft/train.jsonl + val.jsonl (ChatML 格式)
"""

import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from collections import defaultdict


# ============================================================
# 配置
# ============================================================

@dataclass
class BuildConfig:
    """SFT 数据构造配置"""
    total_target: int = 6000          # 目标总量
    min_quality: int = 3              # 最低质量分
    val_ratio: float = 0.1            # 验证集比例
    max_context_tokens: int = 5000    # 最大上下文（user prompt）token 数（中文约 1:1）
    chapter_min_words: int = 800      # 章节最短字数
    chapter_max_words: int = 5000     # 章节最长字数
    continuation_n: int = 2           # 续写任务取前 N-1 章

    # 任务占比
    task_ratios: dict = field(default_factory=lambda: {
        "continuation": 0.30,
        "unit_creation": 0.25,
        "scene_generation": 0.20,
        "style_imitation": 0.15,
        "outline_expansion": 0.10,
    })

    random_seed: int = 42


SYSTEM_PROMPTS = {
    "continuation": "你是一个都市文娱小说作家，擅长轻松诙谐的文风。请根据前文续写下一章。要求：2000-3000字，起承转合，结尾留钩子，三线（事业+感情+日常）并进。",
    "unit_creation": "你是一个都市文娱小说作家。请根据给定的大纲，完成一个完整的叙事单元。包含铺垫、小高潮、收尾，保证节奏紧凑。",
    "scene_generation": "你是一个都市文娱小说作家。请根据场景设定和情境，生成对应的场景内容。语言口语化、有梗、有爽感。",
    "style_imitation": "你是一个都市文娱小说作家。请模仿以下段落的文风和节奏，写一段新的都市文娱场景。",
    "outline_expansion": "你是一个都市文娱小说作家。请根据给定的大纲，扩展为完整的章节内容。",
}


# ============================================================
# 数据加载
# ============================================================

def load_book_data(book_dir: Path) -> dict:
    """加载一本书的所有章节和标注"""
    chapters = {}
    annotations = {}

    # 加载标注
    ann_file = book_dir / "llm_annotations.json"
    if ann_file.exists():
        for a in json.loads(ann_file.read_text('utf-8')):
            annotations[a["chapter_index"]] = a

    # 加载章节
    for ch_file in sorted(book_dir.glob("chapter_*.txt")):
        # 排除 .txt.txt 重复文件和 metadata 等非章节文件
        name = ch_file.stem
        if name.endswith('.txt'):   # chapter_051.txt.txt → stem = chapter_051.txt
            continue
        try:
            ch_num = int(name.split('_')[-1])
        except ValueError:
            continue
        text = ch_file.read_text('utf-8')
        word_count = len(text.replace('\n', '').replace(' ', ''))
        chapters[ch_num] = {
            "text": text,
            "word_count": word_count,
            "annotation": annotations.get(ch_num),
        }

    # 加载 metadata
    meta_file = book_dir / "metadata.json"
    meta = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text('utf-8'))

    return {
        "book_name": book_dir.name,
        "author": meta.get("author", "unknown"),
        "chapters": chapters,
    }


def load_all_books(clean_dir: Path, config: BuildConfig) -> list[dict]:
    """加载所有书籍数据"""
    books = []
    for book_dir in sorted(clean_dir.iterdir()):
        if book_dir.is_dir():
            data = load_book_data(book_dir)
            # 过滤低质量章节（字数过滤对所有任务生效，质量过滤仅对有标注的章节生效）
            valid = {}
            for k, v in data["chapters"].items():
                if not (config.chapter_min_words <= v["word_count"] <= config.chapter_max_words):
                    continue
                ann = v.get("annotation")
                if ann and ann.get("quality_score", 0) < config.min_quality:
                    continue  # 有标注但质量低 → 剔除
                valid[k] = v  # 无标注或标注质量达标 → 保留
            if len(valid) >= 10:  # 至少 10 章有效
                data["chapters"] = valid
                books.append(data)
    return books


# ============================================================
# 任务构造
# ============================================================

def build_continuation_sample(
    book: dict, ch_num: int, config: BuildConfig
) -> Optional[dict]:
    """
    构造章节续写样本
    Input: 前 N-1 章 → Output: 第 N 章
    """
    chapters = book["chapters"]

    # 找到实际的之前章节（处理稀疏编号）
    sorted_keys = sorted(chapters.keys())
    try:
        target_idx = sorted_keys.index(ch_num)
    except ValueError:
        return None

    start_idx = max(0, target_idx - config.continuation_n + 1)
    prev_ch_nums = sorted_keys[start_idx:target_idx]

    if len(prev_ch_nums) < config.continuation_n - 1:
        return None

    # 收集前 N-1 章
    prev_chs = [chapters[i] for i in prev_ch_nums]

    # 构造 prompt：取前情的关键部分
    context_parts = []
    for ch in prev_chs:
        text = ch["text"]
        # 每章取最后 1000 字（最近的上下文）
        if len(text) > 1000:
            text = "..." + text[-1000:]
        context_parts.append(f"【前文片段】\n{text}")

    context = "\n\n".join(context_parts)

    # 控制在最大 token 内
    if len(context) > config.max_context_tokens:
        context = context[-config.max_context_tokens:]

    target_ch = chapters[ch_num]
    instr = f"请根据以上前文，续写第{ch_num}章。要求 2000-3000 字，有起承转合，结尾留悬念。"

    return {
        "task_type": "continuation",
        "system": SYSTEM_PROMPTS["continuation"],
        "user": f"{context}\n\n{instr}",
        "assistant": target_ch["text"],
    }


def build_scene_generation_sample(
    book: dict, ch_num: int, config: BuildConfig
) -> Optional[dict]:
    """
    构造场景生成样本
    Input: 场景标签 + 前 200 字 + 人物情境 → Output: 场景全文
    """
    chapters = book["chapters"]
    ch = chapters.get(ch_num)
    if not ch or not ch.get("annotation"):
        return None

    ann = ch["annotation"]
    scene_type = ann.get("primary_scene", "daily")

    # 场景标签映射到中文名
    scene_names = {
        "daily": "日常互动",
        "release": "作品发布",
        "variety": "综艺录制",
        "business": "商战谈判",
        "romance": "感情戏",
        "face_slapping": "打脸场面",
    }
    scene_cn = scene_names.get(scene_type, scene_type)

    # 取前 200 字作为情境设定
    text = ch["text"]
    context = text[:200] if len(text) > 200 else text

    # 收集已出场人物
    entities = ann.get("entities", {})
    chars_str = ""
    if entities:
        chars = entities.get("characters", entities if isinstance(entities, list) else [])
        if isinstance(chars, list) and chars:
            chars_str = "已出场人物: " + "、".join(str(c) for c in chars[:5])

    instr = (
        f"【场景类型】{scene_cn}\n"
        f"【当前情境】{context}...\n"
        f"{chars_str}\n"
        f"请根据以上设定，展开写一段{scene_cn}场景。"
    )

    return {
        "task_type": "scene_generation",
        "system": SYSTEM_PROMPTS["scene_generation"],
        "user": instr,
        "assistant": text,
    }


def build_style_imitation_sample(
    books: list[dict], config: BuildConfig
) -> Optional[dict]:
    """
    构造风格模仿样本
    同作者随机取 3 段短文作为 few-shot → 另一段为 target
    """
    # 随机选一本书
    book = random.choice(books)
    chapters = list(book["chapters"].values())
    if len(chapters) < 4:
        return None

    # 随机取 4 段（3 段作为 few-shot，1 段作为 target）
    selected = random.sample(chapters, min(4, len(chapters)))
    few_shot = selected[:3]
    target = selected[3]

    # 每段截取 200-400 字
    examples = []
    for i, ch in enumerate(few_shot):
        text = ch["text"]
        start = random.randint(0, max(0, len(text) - 400))
        snippet = text[start:start + 400]
        examples.append(f"【示例{i+1}】\n{snippet}")

    instr = (
        f"作者：{book['author']}\n"
        f"\n"
        + "\n\n".join(examples) +
        f"\n\n请模仿以上{book['author']}的文风和节奏，写一段新的都市文娱场景（2000-3000字）。"
    )

    if len(instr) > config.max_context_tokens:
        # 减少示例
        instr = instr[:config.max_context_tokens]

    return {
        "task_type": "style_imitation",
        "system": SYSTEM_PROMPTS["style_imitation"],
        "user": instr,
        "assistant": target["text"],
    }


def build_outline_expansion_sample(
    book: dict, ch_num: int, config: BuildConfig
) -> Optional[dict]:
    """
    构造大纲扩展样本
    用 LLM 标注中的节奏信息构造简易大纲 → 对应章节内容
    """
    chapters = book["chapters"]
    ch = chapters.get(ch_num)
    if not ch or not ch.get("annotation"):
        return None

    ann = ch["annotation"]
    rhythm = ann.get("rhythm_label", "buildup")
    scene = ann.get("primary_scene", "daily")
    quality = ann.get("quality_score", 3)

    # 用标注信息构造简化大纲
    rhythm_desc = {
        "buildup": "铺垫推进剧情",
        "mini_climax": "小高潮（阶段成果/反转）",
        "major_climax": "大高潮（重大突破）",
        "transition": "过渡舒缓节奏",
        "resolution": "收尾过渡",
    }.get(rhythm, "推进剧情")

    outline = (
        f"章节序号: 第{ch_num}章\n"
        f"本章节奏: {rhythm_desc}\n"
        f"主要场景: {scene}\n"
    )

    # 如果有高潮信息
    if ann.get("climax_type"):
        outline += f"高潮类型: {ann['climax_type']}\n"

    instr = f"【本章大纲】\n{outline}\n请根据以上大纲，展开写完整的章节内容（2000-3000字）。"

    return {
        "task_type": "outline_expansion",
        "system": SYSTEM_PROMPTS["outline_expansion"],
        "user": instr,
        "assistant": ch["text"],
    }


def build_unit_creation_sample(
    book: dict, start_ch: int, end_ch: int, config: BuildConfig
) -> Optional[dict]:
    """
    构造单元创作样本
    利用节奏标注识别单元边界 → 单元大纲为 prompt，单元全文为 target
    """
    chapters = book["chapters"]
    unit_chapters = {}
    for i in range(start_ch, end_ch + 1):
        if i in chapters:
            unit_chapters[i] = chapters[i]

    if len(unit_chapters) < 2:
        return None

    # 用标注信息构造单元大纲
    outline_parts = []
    unit_text_parts = []

    for num, ch in sorted(unit_chapters.items()):
        text = ch["text"]
        ann = ch.get("annotation") or {}
        rhythm = ann.get("rhythm_label", "buildup")
        scene = ann.get("primary_scene", "daily")

        rhythm_marker = {
            "mini_climax": "【小高潮】",
            "major_climax": "【大高潮】",
        }.get(rhythm, "")

        # 取每章前 100 字作为摘要
        preview = text[:100].replace('\n', ' ')
        outline_parts.append(f"第{num}章: {rhythm_marker} {scene} - {preview}...")
        unit_text_parts.append(f"第{num}章\n{text}")

    outline = "【单元大纲】\n" + "\n".join(outline_parts)
    unit_text = "\n\n".join(unit_text_parts)

    if len(unit_text) > 15000:  # 单元太长
        return None

    instr = f"{outline}\n\n请根据以上大纲，写出该单元的完整内容。"

    return {
        "task_type": "unit_creation",
        "system": SYSTEM_PROMPTS["unit_creation"],
        "user": instr,
        "assistant": unit_text,
    }


# ============================================================
# 主流程
# ============================================================

def build_all_samples(
    books: list[dict],
    config: Optional[BuildConfig] = None,
) -> list[dict]:
    """构造所有类型的 SFT 样本"""
    if config is None:
        config = BuildConfig()

    random.seed(config.random_seed)
    samples = []

    for book in books:
        chapters = book["chapters"]
        ch_nums = sorted(chapters.keys())

        # 1. 章节续写
        for ch_num in ch_nums:
            sample = build_continuation_sample(book, ch_num, config)
            if sample:
                samples.append(sample)

        # 2. 场景生成
        for ch_num in ch_nums:
            sample = build_scene_generation_sample(book, ch_num, config)
            if sample:
                samples.append(sample)

        # 3. 大纲扩展
        for ch_num in ch_nums:
            sample = build_outline_expansion_sample(book, ch_num, config)
            if sample:
                samples.append(sample)

        # 4. 单元创作: 用索引滑动窗口取连续 5 个有效章节
        for i in range(0, len(ch_nums) - 4, 2):  # stride=2，单元间有重叠
            start = ch_nums[i]
            end = ch_nums[i + 4]
            sample = build_unit_creation_sample(book, start, end, config)
            if sample:
                samples.append(sample)

    # 5. 风格模仿: 构造与章节数相当的样本
    num_style = len([s for s in samples if s["task_type"] == "continuation"])
    for _ in range(max(50, num_style // 3)):
        sample = build_style_imitation_sample(books, config)
        if sample:
            samples.append(sample)

    # 去重（按 task_type + assistant 前 200 字，避免不同类型任务共用同一章节时误删）
    seen = set()
    unique = []
    for s in samples:
        key = (s["task_type"], s["assistant"][:200])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def format_chatml(sample: dict) -> str:
    """
    格式化为 ChatML 格式（JSONL 存储用字符串）

    注意: 实际训练时需使用 Qwen3 tokenizer.apply_chat_template()
    此处仅做存储格式的标准化，训练脚本会重新解析并用 tokenizer 格式化。
    """
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user", "content": sample["user"]},
        {"role": "assistant", "content": sample["assistant"]},
    ]
    return json.dumps({"messages": messages, "task_type": sample["task_type"]}, ensure_ascii=False)


def balance_and_split(
    samples: list[dict],
    config: BuildConfig,
) -> tuple[list[dict], list[dict]]:
    """按任务类型均衡采样并分割训练/验证集"""
    # 按任务类型分组
    by_type = defaultdict(list)
    for s in samples:
        by_type[s["task_type"]].append(s)

    # 按比例采样
    balanced = []
    for task, ratio in config.task_ratios.items():
        pool = by_type.get(task, [])
        target = int(config.total_target * ratio)
        if len(pool) > target:
            sampled = random.sample(pool, target)
        else:
            sampled = pool
        balanced.extend(sampled)
        print(f"  {task}: {len(sampled)}/{len(pool)} (目标={target})")

    # 如果总量不够，从最大池中补充
    if len(balanced) < config.total_target and by_type:
        max_task = max(by_type, key=lambda t: len(by_type[t]))
        extra = random.sample(
            by_type[max_task],
            min(config.total_target - len(balanced), len(by_type[max_task]))
        )
        balanced.extend(extra)
        print(f"  补充 {max_task}: +{len(extra)}")

    random.shuffle(balanced)

    # 分割
    val_size = max(50, int(len(balanced) * config.val_ratio))
    train = balanced[val_size:]
    val = balanced[:val_size]

    return train, val


def build_sft_dataset(
    clean_dir: Path,
    output_dir: Optional[Path] = None,
    config: Optional[BuildConfig] = None,
) -> tuple[Path, Path]:
    """
    主入口: 构建完整的 SFT 训练数据集

    Returns:
        (train_path, val_path)
    """
    if config is None:
        config = BuildConfig()

    if output_dir is None:
        output_dir = Path.cwd() / "data" / "sft"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("📂 加载数据...")
    books = load_all_books(clean_dir, config)
    total_chapters = sum(len(b["chapters"]) for b in books)
    print(f"  {len(books)} 本书, {total_chapters} 章有效")

    # 2. 构造样本
    print("🔨 构造样本...")
    samples = build_all_samples(books, config)
    print(f"  共构造 {len(samples)} 条样本")

    # 3. 均衡采样
    print("⚖️ 均衡采样...")
    train, val = balance_and_split(samples, config)
    print(f"  train={len(train)}, val={len(val)}")

    # 4. 写入
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, 'w', encoding='utf-8') as f:
            for s in data:
                f.write(format_chatml(s) + '\n')

    print(f"\n✅ SFT 数据集已生成:")
    print(f"  train: {train_path} ({len(train)} 条)")
    print(f"  val:   {val_path} ({len(val)} 条)")

    # 5. 统计
    task_counts = defaultdict(int)
    for s in train:
        task_counts[s["task_type"]] += 1
    print(f"  任务分布: {dict(task_counts)}")

    return train_path, val_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    clean_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "data" / "clean"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "data" / "sft"
    build_sft_dataset(clean_dir, output_dir)
