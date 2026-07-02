"""
共享环境变量加载模块

用法:
    from src.env import load_env, get_api_key
    load_env()
    key = get_api_key("DEEPSEEK_API_KEY")
"""

import os
from pathlib import Path


def _find_project_root() -> Path:
    """向上搜索 .env 或 .git 目录确定项目根目录"""
    current = Path(__file__).resolve().parent.parent
    for marker in [".env", ".git"]:
        marker_path = current / marker
        if marker_path.exists():
            return current
    return current


_PROJECT_ROOT = _find_project_root()
_LOADED = False


def load_env() -> dict[str, str]:
    """
    加载项目根目录的 .env 文件到 os.environ。
    幂等：多次调用只加载一次。
    返回加载的键值对字典。
    """
    global _LOADED
    if _LOADED:
        return {}

    env_file = _PROJECT_ROOT / ".env"
    loaded = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                os.environ.setdefault(k, v)
                loaded[k] = v
    _LOADED = True
    return loaded


def get_api_key(name: str, default: str = "") -> str:
    """获取 API Key（自动加载 .env）"""
    load_env()
    return os.environ.get(name, default)


def get_project_root() -> Path:
    """返回项目根目录"""
    return _PROJECT_ROOT
