"""
数据清洗模块

功能:
- 读取原始 txt 文件，统一编码为 UTF-8
- 去除广告、水印、网站标签
- 章节切分（识别 "第X章" 等模式）
- 过滤过短/无意义段落
- 输出清洗后章节文件 + 基础 metadata
"""

import re
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ChapterMeta:
    """单章元数据"""
    chapter_index: int
    chapter_title: str = ""
    word_count: int = 0
    paragraph_count: int = 0
    start_line: int = 0
    end_line: int = 0


@dataclass
class BookMeta:
    """整书元数据"""
    book_name: str
    author: str = "unknown"
    total_chapters: int = 0
    total_words: int = 0
    encoding: str = "utf-8"
    chapters: list = field(default_factory=list)


# ============================================================
# 配置常量
# ============================================================

# 章节标题正则模式（覆盖主流网文格式）
CHAPTER_PATTERNS = [
    re.compile(r'^[第序][\d一二三四五六七八九十百千万]+[章回节卷集].*'),   # 第X章 / 序章
    re.compile(r'^[Cc]hapter\s*\d+'),                              # Chapter 1
    re.compile(r'^[卷部][\d一二三四五六七八九十百千万]+.*'),          # 第一卷
    re.compile(r'^第[\d一二三四五六七八九十百千万]+[章回].*'),        # 第1章
    re.compile(r'^\d+[、，.\s]+.*'),                               # 1、章节名
    re.compile(r'^[（\(][\d一二三四五六七八九十百千万]+[）\)].*'),    # （一）
]

# 需要过滤的广告/水印模式
AD_PATTERNS = [
    re.compile(r'请记住.*网址'),
    re.compile(r'本书首发.*'),
    re.compile(r'最新章节.*网址'),
    re.compile(r'【.*】'),
    re.compile(r'[（\(]本章未完[）\)].*'),
    re.compile(r'[（\(]未完待续[）\)].*'),
    re.compile(r'ps[：:].*', re.IGNORECASE),
    re.compile(r'作者[说言].*'),
    re.compile(r'如果喜欢.*请.*收藏'),
    re.compile(r'求.*票.*订阅'),
    re.compile(r'more\.\w+\.com', re.IGNORECASE),
    re.compile(r'www\.\w+\.com', re.IGNORECASE),
    re.compile(r'^\s*$'),  # 纯空行（用于过滤空白段落）
]

# 多编码尝试顺序（中文 txt 常见编码）
ENCODING_ORDER = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'utf-16', 'big5', 'latin-1']


# ============================================================
# 编码检测与读取
# ============================================================

def detect_and_read(file_path: Path) -> tuple[str, str]:
    """尝试多种编码读取文件，返回 (文本内容, 实际编码)"""
    raw_bytes = file_path.read_bytes()

    for enc in ENCODING_ORDER:
        try:
            text = raw_bytes.decode(enc)
            # 验证：成功解码且包含中文字符（或至少看起来合理）
            if _looks_like_chinese_novel(text):
                return text, enc
        except (UnicodeDecodeError, LookupError):
            continue

    # 最后的兜底：用 errors='replace' 强制解码
    text = raw_bytes.decode('utf-8', errors='replace')
    return text, 'utf-8 (fallback, lossy)'


def _looks_like_chinese_novel(text: str) -> bool:
    """简单判断文本是否像中文小说"""
    sample = text[:5000]
    chinese_chars = sum(1 for c in sample if '一' <= c <= '鿿')
    total_chars = len(sample.replace('\n', '').replace('\r', '').replace(' ', ''))
    if total_chars == 0:
        return False
    return chinese_chars / total_chars > 0.3


# ============================================================
# 章节检测
# ============================================================

def is_chapter_title(line: str) -> bool:
    """判断一行是否是章节标题"""
    line = line.strip()
    if not line or len(line) > 40:
        return False
    return any(pat.match(line) for pat in CHAPTER_PATTERNS)


def extract_chapter_number(title: str) -> Optional[int]:
    """从章节标题中提取章节序号"""
    # 尝试阿拉伯数字
    m = re.search(r'(\d+)', title)
    if m:
        return int(m.group(1))
    # 尝试中文数字（简单映射，仅支持常见章数）
    cn_num_map = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '百': 100, '千': 1000, '万': 10000,
    }
    # 简化处理：只提取第一个中文数字
    m = re.search(r'([一二三四五六七八九十百千万]+)', title)
    if m:
        cn_str = m.group(1)
        return _parse_chinese_number(cn_str)
    return None


def _parse_chinese_number(s: str) -> int:
    """解析中文数字字符串为整数（简化版，处理 1-9999）"""
    result = 0
    unit = 1
    for i, char in enumerate(reversed(s)):
        if char == '十':
            unit = max(unit, 10)
        elif char == '百':
            unit = max(unit, 100)
        elif char == '千':
            unit = max(unit, 1000)
        elif char == '万':
            unit = max(unit, 10000)
        else:
            digit = {
                '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                '六': 6, '七': 7, '八': 8, '九': 9, '零': 0
            }.get(char, 0)
            result += digit * unit
    if result == 0 and s in ['十', '百', '千', '万']:
        return unit
    return result


# ============================================================
# 文本清洗
# ============================================================

def clean_text(text: str) -> str:
    """清洗文本：统一换行、去除控制字符"""
    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # 去除 BOM
    if text.startswith('﻿'):
        text = text[1:]
    # 去除多余空行（保留单个空行作为段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def is_ad_line(line: str) -> bool:
    """判断是否广告/水印行"""
    line = line.strip()
    if not line:
        return True  # 空行标记为可过滤
    return any(pat.search(line) for pat in AD_PATTERNS)


def filter_paragraph(para: str, min_chars: int = 10) -> bool:
    """判断段落是否应当保留"""
    # 太短的段落过滤
    if len(para.strip()) < min_chars:
        return False
    # 纯符号/数字的段落过滤
    chinese_chars = sum(1 for c in para if '一' <= c <= '鿿')
    if chinese_chars < 3:
        return False
    return True


# ============================================================
# 主流程
# ============================================================

def split_chapters(text: str) -> list[tuple[int, str, str]]:
    """
    将文本按章节切分
    返回: [(章节号, 章节标题, 章节正文), ...]
    """
    lines = text.split('\n')
    chapters = []
    current_chapter_lines = []
    current_title = "楔子"
    current_num = 0

    for line in lines:
        stripped = line.strip()
        if is_chapter_title(stripped):
            # 保存上一个章节
            if current_chapter_lines:
                chapter_text = '\n'.join(current_chapter_lines).strip()
                chapters.append((current_num, current_title, chapter_text))

            # 开始新章节
            current_title = stripped
            current_num = extract_chapter_number(stripped) or (current_num + 1)
            current_chapter_lines = []
        else:
            if not is_ad_line(line):
                current_chapter_lines.append(line)

    # 保存最后一章
    if current_chapter_lines:
        chapter_text = '\n'.join(current_chapter_lines).strip()
        chapters.append((current_num, current_title, chapter_text))

    return chapters


def process_book(
    file_path: Path,
    author: str = "unknown",
    output_dir: Optional[Path] = None,
    min_chapter_words: int = 500,
) -> BookMeta:
    """
    处理单本小说 txt 文件

    Args:
        file_path: 原始 txt 文件路径
        author: 作者名（如果已知）
        output_dir: 输出目录（None 则使用默认路径）
        min_chapter_words: 单章最低字数（低于此值视为无效章）

    Returns:
        BookMeta: 书籍元数据
    """
    # 1. 读取文件
    text, encoding = detect_and_read(file_path)
    text = clean_text(text)

    # 2. 提取书名
    book_name = file_path.stem

    # 3. 章节切分
    raw_chapters = split_chapters(text)

    # 4. 过滤无效章节 + 保存
    meta = BookMeta(
        book_name=book_name,
        author=author,
        encoding=encoding,
        total_chapters=0,
        total_words=0,
    )

    if output_dir is None:
        output_dir = Path.cwd() / "data" / "clean" / book_name

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_chapters = []
    for ch_num, ch_title, ch_text in raw_chapters:
        # 过滤段落
        paragraphs = [p for p in ch_text.split('\n') if filter_paragraph(p)]
        cleaned_text = '\n'.join(paragraphs)
        word_count = len(cleaned_text.replace('\n', '').replace(' ', ''))

        if word_count < min_chapter_words:
            continue

        ch_meta = ChapterMeta(
            chapter_index=ch_num,
            chapter_title=ch_title,
            word_count=word_count,
            paragraph_count=len(paragraphs),
        )
        valid_chapters.append(ch_meta)

        # 写入清洗后的章节文件
        ch_filename = f"chapter_{ch_num:03d}.txt"
        (output_dir / ch_filename).write_text(cleaned_text, encoding='utf-8')

    # 5. 更新元数据
    meta.total_chapters = len(valid_chapters)
    meta.total_words = sum(ch.word_count for ch in valid_chapters)
    meta.chapters = [asdict(ch) for ch in valid_chapters]

    # 6. 保存元数据
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    return meta


def clean_directory(
    input_dir: Path,
    output_base_dir: Optional[Path] = None,
    author_map: Optional[dict[str, str]] = None,
) -> list[BookMeta]:
    """
    批量处理目录下的所有 txt 文件

    Args:
        input_dir: 原始 txt 目录 (data/raw/)
        output_base_dir: 输出根目录 (data/clean/)
        author_map: 文件名 → 作者名的映射

    Returns:
        list[BookMeta]: 所有处理成功的书籍元数据
    """
    if output_base_dir is None:
        output_base_dir = Path.cwd() / "data" / "clean"

    if author_map is None:
        author_map = {}

    results = []
    txt_files = list(input_dir.glob("**/*.txt"))

    for txt_file in txt_files:
        book_name = txt_file.stem
        author = author_map.get(book_name, "unknown")
        try:
            meta = process_book(txt_file, author=author, output_dir=output_base_dir / book_name)
            results.append(meta)
            print(f"✅ {book_name}: {meta.total_chapters} 章, {meta.total_words} 字")
        except Exception as e:
            print(f"❌ {book_name}: {e}")

    print(f"\n总计: {len(results)}/{len(txt_files)} 本处理成功")
    return results


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import sys
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "data" / "raw"
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "data" / "clean"

    if input_path.is_file():
        meta = process_book(input_path)
        print(json.dumps(asdict(meta), ensure_ascii=False, indent=2))
    else:
        clean_directory(input_path, output_path)
