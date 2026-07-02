"""
章节切分与文本清洗
- 编码自动检测 (UTF-8/GBK/GB18030)
- 章节标题识别 (正则匹配)
- 广告/水印行过滤
- 输出: data/clean/{书名}/chapter_XXX.txt + metadata.json
"""

import re, json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ChapterMeta:
    chapter_index: int
    chapter_title: str = ""
    word_count: int = 0
    paragraph_count: int = 0

@dataclass
class BookMeta:
    book_name: str
    author: str = "unknown"
    total_chapters: int = 0
    total_words: int = 0
    encoding: str = "utf-8"
    chapters: list = field(default_factory=list)


CHAPTER_PATTERNS = [
    re.compile(r'^[第序][\d一二三四五六七八九十百千万]+[章回卷].*'),
    re.compile(r'^[Cc]hapter\s*\d+'),
    re.compile(r'^[卷部][\d一二三四五六七八九十百千万]+.*'),
    re.compile(r'^第[\d一二三四五六七八九十百千万]+[章回].*'),
    re.compile(r'^\d+[、]\s*.{2,}'),
    re.compile(r'^[（\(][\d一二三四五六七八九十百千万]+[）\)]\s*.{2,}'),
]

AD_PATTERNS = [
    re.compile(r'请记住.*网址'), re.compile(r'本书首发.*'),
    re.compile(r'最新章节.*网址'), re.compile(r'[（\(]本章未完[）\)].*'),
    re.compile(r'ps[：:].*', re.IGNORECASE), re.compile(r'作者[说言].*'),
    re.compile(r'如果喜欢.*请.*收藏'), re.compile(r'求.*票.*订阅'),
    re.compile(r'more\.\w+\.com', re.IGNORECASE), re.compile(r'www\.\w+\.com', re.IGNORECASE),
]

ENCODING_ORDER = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'utf-16', 'big5', 'latin-1']


def detect_and_read(file_path: Path) -> tuple[str, str]:
    raw_bytes = file_path.read_bytes()
    for enc in ENCODING_ORDER:
        try:
            text = raw_bytes.decode(enc)
            sample = text[:5000]
            chinese = sum(1 for c in sample if '一' <= c <= '鿿')
            total = len(sample.replace('\n', '').replace('\r', '').replace(' ', ''))
            if total > 0 and chinese / total > 0.3:
                return text, enc
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode('utf-8', errors='replace'), 'utf-8 (fallback)'


def is_chapter_title(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 40:
        return False
    if line[-1] in '。！？，、：；…～~）)》」』':
        return False
    if re.search(r'第[一二三四五六七八九十\d百千万]+[章回卷][，、。：；]', line):
        return False
    chinese = sum(1 for c in line if '一' <= c <= '鿿')
    return chinese >= 2 and any(pat.match(line) for pat in CHAPTER_PATTERNS)


def extract_chapter_number(title: str) -> Optional[int]:
    m = re.search(r'第[\s]*([一二三四五六七八九十百千萬零]+)[\s]*[章回卷]', title)
    if m:
        return _parse_cn_number(m.group(1))
    m = re.search(r'第[\s]*(\d+)[\s]*[章回卷]', title)
    if m:
        return int(m.group(1))
    m = re.search(r'^[\s]*(\d+)[\s]*[、]', title)
    if m:
        return int(m.group(1))
    m = re.search(r'([一二三四五六七八九十百千万]+)', title)
    if m:
        return _parse_cn_number(m.group(1))
    return None


def _parse_cn_number(s: str) -> int:
    digit_map = {'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    unit_map = {'十':10,'百':100,'千':1000,'万':10000}
    result, current = 0, 0
    for char in s:
        if char in digit_map:
            current = digit_map[char]
        elif char in unit_map:
            unit = unit_map[char]
            if current == 0:
                current = 1
            current *= unit
            if unit >= 10:
                result += current
                current = 0
    result += current
    return result if result > 0 else 1


def is_ad_line(line: str) -> bool:
    line = line.strip()
    return not line or any(pat.search(line) for pat in AD_PATTERNS)


def filter_paragraph(para: str, min_chars: int = 10) -> bool:
    if len(para.strip()) < min_chars:
        return False
    chinese = sum(1 for c in para if '一' <= c <= '鿿')
    return chinese >= 3


def split_chapters(text: str) -> list[tuple[int, str, str]]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    if text.startswith('﻿'):
        text = text[1:]
    text = re.sub(r'\n{3,}', '\n\n', text)

    lines = text.split('\n')
    chapters = []
    current_lines, current_title, current_num = [], "楔子", 0

    for line in lines:
        stripped = line.strip()
        if is_chapter_title(stripped):
            if current_lines:
                chapters.append((current_num, current_title, '\n'.join(current_lines).strip()))
            current_title = stripped
            current_num = extract_chapter_number(stripped) or (current_num + 1)
            current_lines = []
        elif not is_ad_line(line):
            current_lines.append(line)

    if current_lines:
        chapters.append((current_num, current_title, '\n'.join(current_lines).strip()))
    return chapters


def process_book(file_path: Path, author: str = "unknown",
                 output_dir: Optional[Path] = None, min_chapter_words: int = 500) -> BookMeta:
    text, encoding = detect_and_read(file_path)
    book_name = file_path.stem
    raw_chapters = split_chapters(text)

    if output_dir is None:
        output_dir = Path.cwd() / "data" / "clean" / book_name
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = BookMeta(book_name=book_name, author=author, encoding=encoding)
    valid_chapters = []

    for ch_num, ch_title, ch_text in raw_chapters:
        paragraphs = [p for p in ch_text.split('\n') if filter_paragraph(p)]
        cleaned = '\n'.join(paragraphs)
        wc = len(cleaned.replace('\n', '').replace(' ', ''))
        if wc < min_chapter_words:
            continue

        valid_chapters.append(ChapterMeta(chapter_index=ch_num, chapter_title=ch_title,
                                           word_count=wc, paragraph_count=len(paragraphs)))
        (output_dir / f"chapter_{ch_num:03d}.txt").write_text(cleaned, encoding='utf-8')

    meta.total_chapters = len(valid_chapters)
    meta.total_words = sum(ch.word_count for ch in valid_chapters)
    meta.chapters = [asdict(ch) for ch in valid_chapters]
    (output_dir / "metadata.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding='utf-8')
    return meta


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "data" / "raw"
    if p.is_file():
        process_book(p)
    else:
        for f in p.glob("**/*.txt"):
            process_book(f)
