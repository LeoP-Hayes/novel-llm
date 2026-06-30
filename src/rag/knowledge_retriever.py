"""
知识检索模块

管理结构化知识库（人物/作品/世界观）和语义检索（高潮章节/黄金三章），
为章节生成提供上下文一致性保障。
"""

import json
from pathlib import Path
from typing import Optional

from .embedding import EmbeddingEngine, ChromaManager


# ============================================================
# 结构化知识库
# ============================================================

class KnowledgeBase:
    """结构化知识库（JSON 文件存储）"""

    def __init__(self, kb_dir: Path = Path("data/knowledge")):
        kb_dir.mkdir(parents=True, exist_ok=True)
        self.kb_dir = kb_dir
        self._characters_file = kb_dir / "characters.json"
        self._works_file = kb_dir / "works.json"
        self._world_file = kb_dir / "world_settings.json"
        self._industry_file = kb_dir / "industry_kb.json"

    # --- 人物 ---
    @property
    def characters(self) -> list[dict]:
        if self._characters_file.exists():
            return json.loads(self._characters_file.read_text("utf-8"))
        return []

    def add_character(self, char: dict):
        chars = self.characters
        # 去重
        if not any(c.get("name") == char.get("name") for c in chars):
            chars.append(char)
            self._characters_file.write_text(json.dumps(chars, ensure_ascii=False, indent=2))

    def update_character(self, name: str, updates: dict):
        chars = self.characters
        for c in chars:
            if c.get("name") == name:
                c.update(updates)
                break
        self._characters_file.write_text(json.dumps(chars, ensure_ascii=False, indent=2))

    def get_character(self, name: str) -> Optional[dict]:
        for c in self.characters:
            if c.get("name") == name:
                return c
        return None

    def format_characters(self) -> str:
        """格式化人物表为 Prompt 片段"""
        if not self.characters:
            return "暂无"
        parts = []
        for c in self.characters:
            parts.append(
                f"  {c.get('name','?')}: {c.get('role','?')} | "
                f"{c.get('career','?')} | {c.get('traits','')} | "
                f"金手指: {c.get('golden_finger','无')}"
            )
        return "\n".join(parts)

    # --- 作品 ---
    @property
    def works(self) -> list[dict]:
        if self._works_file.exists():
            return json.loads(self._works_file.read_text("utf-8"))
        return []

    def add_work(self, work: dict):
        works = self.works
        if not any(w.get("title") == work.get("title") for w in works):
            works.append(work)
            self._works_file.write_text(json.dumps(works, ensure_ascii=False, indent=2))

    # --- 世界观 ---
    @property
    def world_settings(self) -> dict:
        if self._world_file.exists():
            return json.loads(self._world_file.read_text("utf-8"))
        return {}

    def set_world(self, settings: dict):
        self._world_file.write_text(json.dumps(settings, ensure_ascii=False, indent=2))

    def format_world(self) -> str:
        if not self.world_settings:
            return "暂无"
        s = self.world_settings
        return f"时代: {s.get('era','?')} | 背景: {s.get('industry','?')} | 平行世界: {s.get('parallel',False)}"


# ============================================================
# 语义知识检索
# ============================================================

class KnowledgeRetriever:
    """知识检索器（语义 + 结构化）"""

    def __init__(self, kb_dir: Path = Path("data/knowledge")):
        self.kb = KnowledgeBase(kb_dir)
        self.embedder = EmbeddingEngine()
        self.chroma = ChromaManager()
        self.collection = self.chroma.get_or_create_collection("knowledge_index")

    def retrieve_similar_scenes(self, query: str, top_k: int = 3) -> list[dict]:
        """检索相似的知识密集场景"""
        query_vec = self.embedder.encode_query(query)
        return self.chroma.search(self.collection, query_vec, top_k=top_k)

    def retrieve_golden_three(self) -> list[dict]:
        """检索黄金三章的知识片段"""
        return self.chroma.search(
            self.collection,
            query_embedding=self.embedder.encode_query("人物登场 金手指激活 世界设定"),
            top_k=10,
            where={"is_golden": True},
        )

    def initialize_from_outline(self, outline: dict):
        """从大纲初始化冷启动知识库"""
        # 从大纲提取人物
        for unit in outline.get("units", []):
            for line_name in ["career", "romance", "daily"]:
                # 简单的关键词提取
                pass  # 实际使用时由 LLM 辅助提取

        # 设置世界观
        self.kb.set_world({
            "era": outline.get("era", ""),
            "industry": "娱乐圈",
            "parallel": outline.get("parallel", True),
        })

    def format_knowledge_context(self) -> str:
        """格式化为 Prompt 中的知识上下文"""
        return (
            f"【世界观】{self.kb.format_world()}\n"
            f"【人物】\n{self.kb.format_characters()}"
        )


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_character({
        "name": "林风", "role": "主角", "career": "导演",
        "golden_finger": "重生2008", "traits": "幽默、腹黑",
    })
    kb.add_character({
        "name": "苏晚晴", "role": "女主", "career": "演员",
        "golden_finger": "无", "traits": "清冷、有实力",
    })
    kb.set_world({"era": "2008-2015", "industry": "娱乐圈", "parallel": True})

    print("=== 人物表 ===")
    print(kb.format_characters())
    print()
    print("=== 世界观 ===")
    print(kb.format_world())

    retriever = KnowledgeRetriever()
    print("\n=== 黄金三章检索 ===")
    results = retriever.retrieve_golden_three()
    print(f"  找到 {len(results)} 条")
