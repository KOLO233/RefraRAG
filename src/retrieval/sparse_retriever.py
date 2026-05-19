"""Sparse Retriever — BM25 稀疏检索器。

基于 BM25 算法的关键词检索，解决专有名词精确匹配问题。
与 Dense Retriever 互补：Dense 理解语义，Sparse 精确匹配关键词。

BM25 统计数据由 EmbeddingService 管理，本模块只负责检索逻辑。
参考 SuperMew 的 BM25 实现思路。
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import List, Optional

from src.core.types import RetrievalResult
from src.libs.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class SparseRetriever:
    """BM25 稀疏检索器。

    流程：
    1. 查询分词
    2. 对每个文档计算 BM25 分数
    3. 按分数降序返回 Top-K

    注意：当前实现需要遍历 Milvus 中的所有文档进行 BM25 打分。
    对于小规模文档库（<10万条）性能可接受；大规模场景需要建立倒排索引。

    Example:
        >>> retriever = SparseRetriever(embedding_service, milvus_store)
        >>> results = await retriever.retrieve(keywords=["机器", "学习"], top_k=5)
    """

    def __init__(self, embedding_service: EmbeddingService, milvus_store=None):
        self._embedding = embedding_service
        self._store = milvus_store

    async def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        query_text: str = "",
    ) -> List[RetrievalResult]:
        """执行 BM25 稀疏检索。

        Args:
            keywords: 查询关键词列表
            top_k: 返回的最大结果数
            query_text: 原始查询文本（用于分词，如果 keywords 为空）

        Returns:
            RetrievalResult 列表，按 BM25 分数降序排列
        """
        # 如果没有关键词，从查询文本中提取
        if not keywords and query_text:
            keywords = self._embedding.tokenize(query_text)

        if not keywords:
            return []

        # 从 Milvus 获取所有文档（用于 BM25 打分）
        if self._store is None:
            logger.warning("No Milvus store configured for sparse retrieval")
            return []

        try:
            # 获取所有文档的文本
            all_docs = self._store._get_client().query(
                self._store._collection,
                output_fields=["chunk_id", "text", "filename", "source_path",
                               "page", "chunk_index", "chunk_level",
                               "parent_chunk_id", "root_chunk_id"],
                limit=16384,
            )
        except Exception as e:
            logger.error(f"Failed to query documents for BM25: {e}")
            return []

        if not all_docs:
            return []

        # 对每个文档计算 BM25 分数
        query_tokens = self._embedding.tokenize(" ".join(keywords))
        scored_docs = []

        for doc in all_docs:
            text = doc.get("text", "")
            if not text:
                continue

            doc_tokens = self._embedding.tokenize(text)
            score = self._embedding.compute_bm25_score(query_tokens, doc_tokens)

            if score > 0:
                scored_docs.append((doc, score))

        # 按分数降序排列
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 转换为 RetrievalResult
        results = []
        for doc, score in scored_docs[:top_k]:
            result = RetrievalResult(
                chunk_id=doc.get("chunk_id", ""),
                score=score,
                text=doc.get("text", ""),
                metadata={
                    "source_path": doc.get("source_path", ""),
                    "filename": doc.get("filename", ""),
                    "page": doc.get("page", 0),
                    "chunk_index": doc.get("chunk_index", 0),
                    "chunk_level": doc.get("chunk_level", 3),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                },
                retrieval_source="sparse",
            )
            results.append(result)

        logger.debug(
            f"Sparse retrieval: keywords={keywords[:5]}, "
            f"docs_scanned={len(all_docs)}, results={len(results)}"
        )
        return results
