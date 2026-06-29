"""
数据采集模块

功能:
- 批量下载网文 txt 文件（HTTP 直链、网盘链接）
- 格式检测（验证是否为有效的小说 txt）
- 去重（基于文件名 + 文件哈希）
- 组织到 data/raw/ 目录

注意:
- 本模块仅用于下载已公开分享的 txt 资源
- 不支持绕过付费墙或破解付费内容
- 仅供个人学习研究使用
"""

import hashlib
import re
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

import httpx


# ============================================================
# 配置
# ============================================================

# 合法的 txt 来源 URL 模式（公开资源站）
ALLOWED_DOMAINS = [
    "github.com",
    "raw.githubusercontent.com",
    "gist.githubusercontent.com",
]

# 文件名非法字符
ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*]')

# 最小文件大小（有效小说通常 > 50KB）
MIN_FILE_SIZE_KB = 50

# 最大文件大小（单本小说 < 50MB）
MAX_FILE_SIZE_MB = 50


# ============================================================
# 文件哈希与去重
# ============================================================

def compute_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """计算文件哈希，用于去重"""
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def is_duplicate(file_path: Path, existing_hashes: set[str]) -> bool:
    """检查文件是否与已有文件重复"""
    file_hash = compute_file_hash(file_path)
    if file_hash in existing_hashes:
        return True
    existing_hashes.add(file_hash)
    return False


# ============================================================
# 文件验证
# ============================================================

def is_valid_novel_txt(file_path: Path) -> tuple[bool, str]:
    """
    验证文件是否为有效的小说 txt

    返回: (是否有效, 原因描述)
    """
    # 大小检查
    size_kb = file_path.stat().st_size / 1024
    if size_kb < MIN_FILE_SIZE_KB:
        return False, f"文件过小 ({size_kb:.0f}KB < {MIN_FILE_SIZE_KB}KB)"

    size_mb = size_kb / 1024
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"文件过大 ({size_mb:.0f}MB > {MAX_FILE_SIZE_MB}MB)"

    # 编码检查
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(5000)
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                sample = f.read(5000)
        except UnicodeDecodeError:
            return False, "无法解码（非 UTF-8 或 GBK）"

    # 内容检查：中文比例
    chinese_chars = sum(1 for c in sample if '一' <= c <= '鿿')
    total_non_space = len(sample.replace('\n', '').replace('\r', '').replace(' ', ''))
    if total_non_space == 0:
        return False, "文件为空"
    chinese_ratio = chinese_chars / total_non_space
    if chinese_ratio < 0.2:
        return False, f"中文比例过低 ({chinese_ratio:.1%})"

    # 章节检测：是否有至少 3 个章节标题
    chapter_pattern = re.compile(r'第[\d一二三四五六七八九十百千]+章')
    chapters = chapter_pattern.findall(sample)
    # 在更大范围内检测
    if len(chapters) < 2:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_sample = f.read(50000)
        chapters = chapter_pattern.findall(full_sample)

    if len(chapters) < 3:
        return False, f"章节数不足 ({len(chapters)} 章)"

    return True, f"有效 ({size_kb:.0f}KB, {len(chapters)}+ 章)"


# ============================================================
# 下载与组织
# ============================================================

def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    return ILLEGAL_CHARS.sub('_', name).strip()


def download_file(
    url: str,
    output_dir: Path,
    filename: Optional[str] = None,
    timeout: int = 60,
) -> Optional[Path]:
    """
    下载单个 txt 文件

    Args:
        url: 下载链接
        output_dir: 输出目录
        filename: 指定文件名（None 则从 URL 提取）
        timeout: 超时秒数

    Returns:
        下载成功返回文件路径，失败返回 None
    """
    if filename is None:
        # 从 URL 提取文件名
        parsed = urlparse(url)
        filename = unquote(Path(parsed.path).name)
        if not filename.endswith('.txt'):
            filename += '.txt'

    filename = sanitize_filename(filename)
    output_path = output_dir / filename

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

            output_path.write_bytes(response.content)

            # 验证
            valid, reason = is_valid_novel_txt(output_path)
            if not valid:
                output_path.unlink()
                print(f"⚠️ {filename}: 验证失败 - {reason}")
                return None

            print(f"✅ {filename}: 下载成功 ({output_path.stat().st_size / 1024:.0f}KB)")
            return output_path

    except Exception as e:
        print(f"❌ {filename}: 下载失败 - {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def organize_raw_files(
    source_dir: Path,
    output_dir: Path,
    author_map: Optional[dict[str, str]] = None,
) -> dict[str, Path]:
    """
    将已有 txt 文件按 {作者}/{书名}.txt 结构组织到 data/raw/

    Args:
        source_dir: 包含 txt 文件的源目录
        output_dir: 输出目录 (data/raw/)
        author_map: 文件名 → 作者名的映射

    Returns:
        文件名 → 目标路径的映射
    """
    if author_map is None:
        author_map = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_hashes: set[str] = set()

    # 先收集已有文件的哈希
    for existing in output_dir.glob("**/*.txt"):
        existing_hashes.add(compute_file_hash(existing))

    results = {}
    txt_files = list(source_dir.glob("**/*.txt"))

    for txt_file in txt_files:
        # 验证
        valid, reason = is_valid_novel_txt(txt_file)
        if not valid:
            print(f"⚠️ {txt_file.name}: 跳过 - {reason}")
            continue

        # 去重
        if is_duplicate(txt_file, existing_hashes):
            print(f"⏭️ {txt_file.name}: 重复文件，跳过")
            continue

        # 组织
        book_name = txt_file.stem
        author = author_map.get(book_name, "unknown")
        target_dir = output_dir / author
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{book_name}.txt"

        shutil.copy2(txt_file, target_path)
        results[book_name] = target_path
        print(f"✅ {txt_file.name} → {target_path}")

    print(f"\n总计: {len(results)} 本新书入库")
    return results


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python collector.py <源目录或URL> [输出目录]")
        sys.exit(1)

    source = sys.argv[1]
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "data" / "raw"

    if source.startswith("http"):
        download_file(source, output)
    else:
        organize_raw_files(Path(source), output)
