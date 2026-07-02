"""
数据采集模块
- HTTP 下载 / 本地 txt 组织到 data/raw/
- 文件验证 (大小/编码/中文比例/章节数量)
- MD5 去重
"""

import hashlib, re, shutil
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote
import httpx

MIN_FILE_SIZE_KB = 50
MAX_FILE_SIZE_MB = 50
ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*]')


def compute_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def is_valid_novel_txt(file_path: Path) -> tuple[bool, str]:
    size_kb = file_path.stat().st_size / 1024
    size_mb = size_kb / 1024
    if size_kb < MIN_FILE_SIZE_KB:
        return False, f"文件过小 ({size_kb:.0f}KB)"
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"文件过大 ({size_mb:.0f}MB)"

    for enc in ['utf-8', 'gbk']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                sample = f.read(50000)
            break
        except UnicodeDecodeError:
            continue
    else:
        return False, "无法解码"

    chinese = sum(1 for c in sample if '一' <= c <= '鿿')
    total = len(sample.replace('\n', '').replace('\r', '').replace(' ', ''))
    if total == 0 or chinese / total < 0.2:
        return False, f"中文比例过低"

    chapters = len(re.findall(r'第[\d一二三四五六七八九十百千]+章', sample))
    if chapters < 3:
        return False, f"章节数不足 ({chapters})"

    return True, f"有效 ({size_kb:.0f}KB, {chapters}+章)"


def sanitize_filename(name: str) -> str:
    return ILLEGAL_CHARS.sub('_', name).strip()


def download_file(url: str, output_dir: Path, filename: Optional[str] = None,
                  timeout: int = 60) -> Optional[Path]:
    if filename is None:
        filename = sanitize_filename(unquote(Path(urlparse(url).path).name))
        if not filename.endswith('.txt'):
            filename += '.txt'
    output_path = output_dir / filename
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            valid, reason = is_valid_novel_txt(output_path)
            if not valid:
                output_path.unlink()
                print(f"⚠️ {filename}: {reason}")
                return None
            print(f"✅ {filename}: {output_path.stat().st_size/1024:.0f}KB")
            return output_path
    except Exception as e:
        print(f"❌ {filename}: {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def organize_raw_files(source_dir: Path, output_dir: Path,
                       author_map: Optional[dict[str, str]] = None) -> dict[str, Path]:
    if author_map is None:
        author_map = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = set()
    for f in output_dir.glob("**/*.txt"):
        existing.add(compute_file_hash(f))

    results = {}
    for txt_file in source_dir.glob("**/*.txt"):
        valid, reason = is_valid_novel_txt(txt_file)
        if not valid:
            print(f"⚠️ {txt_file.name}: {reason}")
            continue
        h = compute_file_hash(txt_file)
        if h in existing:
            print(f"⏭️ {txt_file.name}: 重复")
            continue
        existing.add(h)

        book_name = txt_file.stem
        author = author_map.get(book_name, "unknown")
        target_dir = output_dir / author
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{book_name}.txt"
        shutil.copy2(txt_file, target)
        results[book_name] = target
        print(f"✅ {txt_file.name} → {target}")

    print(f"\n总计: {len(results)} 本新书入库")
    return results


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "data" / "raw"
    if source.startswith("http"):
        download_file(source, output)
    else:
        organize_raw_files(Path(source), output)
