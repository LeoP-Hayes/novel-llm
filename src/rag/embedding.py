"""
文本向量化模块

使用 BAAI/bge-large-zh-v1.5 将网文章节段落编码为向量，
存入 ChromaDB，供文风检索和知识检索使用。
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer


# ============================================================
# 配置
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
CHROMA_PERSIST_DIR = "./data/rag/chroma_db"
CHUNK_SIZE = 400       # 每段最大字数
CHUNK_OVERLAP = 50     # 段落间重叠字数


# ============================================================
# Embedding 引擎
# ============================================================

class EmbeddingEngine:
    """BGE 中文 Embedding 引擎"""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本为向量"""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def encode_query(self, text: str) -> list[float]:
        """编码查询文本（单条）"""
        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()


# ============================================================
# ChromaDB 管理
# ============================================================

class ChromaManager:
    """ChromaDB 向量库管理器"""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """获取或创建集合"""
        return self.client.get_or_create_collection(name=name)

    def add_chunks(
        self,
        collection: chromadb.Collection,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ):
        """批量添加 chunk 到向量库"""
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                embeddings=embeddings[i:end],
                documents=chunks[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )

    def search(
        self,
        collection: chromadb.Collection,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """向量检索"""
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["documents"][0]))
        ]


# ============================================================
# 章节分段
# ============================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """将长文本切分为重叠段落"""
    text = text.replace('\n', ' ').strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # 尽量在句号处断开
        if end < len(text):
            for sep in '。！？':
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + 1
                    break
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ============================================================
# 索引构建主流程
# ============================================================

def build_style_index(
    clean_dir: Path,
    embedder: Optional[EmbeddingEngine] = None,
    chroma: Optional[ChromaManager] = None,
    batch_size: int = 200,
):
    """构建文风检索向量索引（分批编码，控制内存）"""
    if embedder is None:
        embedder = EmbeddingEngine()
    if chroma is None:
        chroma = ChromaManager()

    collection = chroma.get_or_create_collection("style_index")

    # 1. 收集所有 chunk 元数据（不编码）
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_id = 0

    for book_dir in sorted(clean_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        ann_file = book_dir / "llm_annotations.json"
        annotations = {}
        if ann_file.exists():
            for a in json.loads(ann_file.read_text("utf-8")):
                annotations[a["chapter_index"]] = a

        meta_file = book_dir / "metadata.json"
        author = "unknown"
        if meta_file.exists():
            author = json.loads(meta_file.read_text("utf-8")).get("author", "unknown")

        for ch_file in sorted(book_dir.glob("chapter_*.txt")):
            ch_num = int(ch_file.stem.split('_')[-1])
            text = ch_file.read_text("utf-8")
            ann = annotations.get(ch_num, {})

            chunks = chunk_text(text)
            for c in chunks:
                all_chunks.append(c)
                all_metadatas.append({
                    "book": book_dir.name,
                    "author": author,
                    "chapter": ch_num,
                    "scene_type": ann.get("primary_scene", ""),
                    "rhythm": ann.get("rhythm_label", ""),
                    "quality": ann.get("quality_score", 0),
                    "climax_type": ann.get("climax_type", ""),
                })
                all_ids.append(f"{book_dir.name}_ch{ch_num}_{chunk_id}")
                chunk_id += 1

    print(f"📊 共 {len(all_chunks)} 个段落，分 {len(all_chunks)//batch_size + 1} 批编码")

    # 2. 分批编码 + 写入
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        batch_chunks = all_chunks[i:end]
        batch_embs = embedder.encode(batch_chunks)
        chroma.add_chunks(
            collection, batch_chunks, batch_embs,
            all_metadatas[i:end], all_ids[i:end],
        )
        print(f"  [{end}/{len(all_chunks)}] 已写入")

    print(f"✅ 文风索引构建完成: {len(all_chunks)} 个向量, dim={embedder.dim}")
    return collection


def build_knowledge_index(
    clean_dir: Path,
    embedder: Optional[EmbeddingEngine] = None,
    chroma: Optional[ChromaManager] = None,
    batch_size: int = 200,
):
    """构建知识检索向量索引（分批编码）"""
    if embedder is None:
        embedder = EmbeddingEngine()
    if chroma is None:
        chroma = ChromaManager()

    collection = chroma.get_or_create_collection("knowledge_index")

    all_chunks = []
    all_metadatas = []
    all_ids = []

    for book_dir in sorted(clean_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        ann_file = book_dir / "llm_annotations.json"
        annotations = {}
        if ann_file.exists():
            for a in json.loads(ann_file.read_text("utf-8")):
                annotations[a["chapter_index"]] = a

        for ch_file in sorted(book_dir.glob("chapter_*.txt")):
            ch_num = int(ch_file.stem.split('_')[-1])
            text = ch_file.read_text("utf-8")
            ann = annotations.get(ch_num, {})

            is_climax = ann.get("rhythm_label") in ("mini_climax", "major_climax")
            is_golden = ch_num <= 3
            if not (is_climax or is_golden):
                continue

            chunks = chunk_text(text, chunk_size=300)
            for c in chunks:
                all_chunks.append(c)
                all_metadatas.append({
                    "book": book_dir.name,
                    "chapter": ch_num,
                    "scene_type": ann.get("primary_scene", ""),
                    "rhythm": ann.get("rhythm_label", ""),
                    "is_golden": is_golden,
                })
                all_ids.append(f"kb_{book_dir.name}_ch{ch_num}_{len(all_ids)}")

    print(f"📊 共 {len(all_chunks)} 个知识片段")

    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        batch_chunks = all_chunks[i:end]
        batch_embs = embedder.encode(batch_chunks)
        chroma.add_chunks(
            collection, batch_chunks, batch_embs,
            all_metadatas[i:end], all_ids[i:end],
        )
        if (i // batch_size) % 5 == 0:
            print(f"  [{end}/{len(all_chunks)}] 已写入")

    print(f"✅ 知识索引构建完成: {len(all_chunks)} 个向量")
    return collection


if __name__ == "__main__":
    import sys
    clean_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "data" / "clean"

    print("=" * 50)
    print("构建文风检索索引...")
    print("=" * 50)
    build_style_index(clean_dir)

    print()
    print("=" * 50)
    print("构建知识检索索引...")
    print("=" * 50)
    build_knowledge_index(clean_dir)
