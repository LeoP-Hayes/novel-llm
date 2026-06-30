"""
文风检索模块

根据场景描述查询最匹配的都市文娱段落，
支持按场景类型/节奏标注/作者等元数据过滤。
"""

from pathlib import Path
from typing import Optional

from .embedding import EmbeddingEngine, ChromaManager


class StyleRetriever:
    """文风检索器"""

    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.chroma = ChromaManager()
        self.collection = self.chroma.get_or_create_collection("style_index")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        scene_filter: Optional[str] = None,
        rhythm_filter: Optional[str] = None,
        author_filter: Optional[str] = None,
        min_quality: int = 3,
    ) -> list[dict]:
        """
        检索相似文风段落

        Args:
            query: 场景描述或写作意图
            top_k: 返回结果数
            scene_filter: 限定场景类型 (daily/release/variety/business/romance/face_slapping)
            rhythm_filter: 限定节奏标注 (buildup/mini_climax/major_climax/transition/resolution)
            author_filter: 限定作者
            min_quality: 最低质量分

        Returns:
            [{"text": ..., "metadata": {...}, "distance": ...}, ...]
        """
        # 构建过滤条件
        where = {"quality": {"$gte": min_quality}}
        if scene_filter:
            where["scene_type"] = scene_filter
        if rhythm_filter:
            where["rhythm"] = rhythm_filter
        if author_filter:
            where["author"] = author_filter

        query_vec = self.embedder.encode_query(query)
        results = self.chroma.search(self.collection, query_vec, top_k=top_k, where=where)
        return results

    def retrieve_face_slapping(self, query: str, top_k: int = 3) -> list[dict]:
        """专门检索打脸名场面"""
        return self.retrieve(query, top_k=top_k, scene_filter="face_slapping")

    def retrieve_climax(self, query: str, top_k: int = 3) -> list[dict]:
        """专门检索高潮段落"""
        return self.retrieve(query, top_k=top_k, rhythm_filter="mini_climax")

    def retrieve_by_author(self, query: str, author: str, top_k: int = 3) -> list[dict]:
        """按作者检索"""
        return self.retrieve(query, top_k=top_k, author_filter=author)

    def format_for_prompt(self, results: list[dict]) -> str:
        """格式化为 Prompt 中的文风参考"""
        parts = []
        for i, r in enumerate(results):
            meta = r["metadata"]
            parts.append(
                f"【参考{i+1}】{meta.get('book','')} 第{meta.get('chapter','')}章 "
                f"({meta.get('scene_type','')}/{meta.get('rhythm','')})\n{r['text'][:300]}"
            )
        return "\n\n".join(parts)


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    retriever = StyleRetriever()

    queries = [
        ("主角新歌发布后全网震惊，热搜第一", "release"),
        ("投资人对主角的导演能力表示质疑，主角冷笑", "face_slapping"),
        ("剧组日常，几个演员在片场互开玩笑", "daily"),
    ]

    for query, scene in queries:
        print(f"\n🔍 {query}")
        results = retriever.retrieve(query, top_k=2, scene_filter=scene)
        for r in results:
            print(f"  [{r['metadata'].get('book','')} ch{r['metadata'].get('chapter','')}] "
                  f"dist={r['distance']:.3f} | {r['text'][:80]}...")
