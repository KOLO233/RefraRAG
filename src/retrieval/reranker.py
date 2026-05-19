"""重排序器。

两段式检索架构的精排阶段：粗排（Dense+Sparse+RRF）→ 精排（Reranker）。

支持两种后端：
1. Cross-Encoder：本地模型，精度高但需要 PyTorch
2. LLM Rerank：用 LLM API 打分，不需要本地模型

当 Cross-Encoder 不可用时自动降级为 LLM Rerank。
"""

from __future__ import annotations

import logging
from typing import List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)

RERANK_PROMPT = """你是一个文档相关性评估专家。
请对以下每个文档片段与查询的相关性进行打分（0-10分）。

查询：{query}

文档片段：
{documents}

请仅输出一个 JSON 数组，包含每个文档的分数，顺序与输入一致。
例如：[8, 5, 9, 3, 7]
不要输出其他任何内容。
"""


class Reranker:
    """重排序器。

    自动选择可用的后端：Cross-Encoder > LLM Rerank > 直接返回。

    Example:
        >>> reranker = Reranker(llm_service=llm)
        >>> reranked = await reranker.rerank("什么是机器学习？", results, top_k=5)
    """

    def __init__(
        self,
        llm_service=None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self._llm = llm_service
        self._model_name = model_name
        self._cross_encoder = None
        self._backend = "auto"  # auto / cross_encoder / llm

    def _try_load_cross_encoder(self) -> bool:
        """尝试加载 Cross-Encoder 模型。"""
        if self._cross_encoder is not None:
            return True
        # 检查 sentence_transformers 是否可用（避免 PyTorch 崩溃）
        import importlib
        try:
            spec = importlib.util.find_spec("sentence_transformers")
            if spec is None:
                return False
        except Exception:
            return False
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._model_name)
            logger.info(f"Loaded Cross-Encoder: {self._model_name}")
            return True
        except Exception as e:
            logger.debug(f"Cross-Encoder not available: {e}")
            return False

    async def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """对检索结果重排序。

        Args:
            query: 查询文本
            results: 待重排的检索结果
            top_k: 返回的最大结果数

        Returns:
            重排后的 RetrievalResult 列表
        """
        if not results:
            return []

        # 尝试 Cross-Encoder（仅在明确指定时）
        if self._backend == "cross_encoder" and self._try_load_cross_encoder():
            return self._cross_encoder_rerank(query, results, top_k)

        # LLM Rerank
        if self._llm:
            return await self._llm_rerank(query, results, top_k)

        # 最终降级：直接返回
        logger.warning("No reranker available, returning original order")
        return results[:top_k]

    def _cross_encoder_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Cross-Encoder 重排序。"""
        pairs = [(query, r.text[:512]) for r in results]
        scores = self._cross_encoder.predict(pairs)

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored[:top_k]:
            reranked.append(RetrievalResult(
                chunk_id=result.chunk_id,
                score=float(score),
                text=result.text,
                metadata={**result.metadata, "rerank_score": float(score)},
                retrieval_source=result.retrieval_source,
            ))

        logger.debug(f"Cross-Encoder reranked {len(results)} → {len(reranked)}")
        return reranked

    async def _llm_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """LLM 重排序（降级方案）。"""
        # 构造文档列表
        doc_lines = []
        for i, r in enumerate(results):
            snippet = r.text[:200].replace("\n", " ")
            doc_lines.append(f"[{i}] {snippet}")

        prompt = RERANK_PROMPT.format(
            query=query,
            documents="\n".join(doc_lines),
        )

        try:
            response = await self._llm.ainvoke(prompt)
            scores = self._parse_scores(response, len(results))

            scored = list(zip(results, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for result, score in scored[:top_k]:
                # 归一化到 0-1
                norm_score = score / 10.0
                reranked.append(RetrievalResult(
                    chunk_id=result.chunk_id,
                    score=norm_score,
                    text=result.text,
                    metadata={**result.metadata, "rerank_score": norm_score},
                    retrieval_source=result.retrieval_source,
                ))

            logger.debug(f"LLM reranked {len(results)} → {len(reranked)}")
            return reranked

        except Exception as e:
            logger.error(f"LLM rerank failed: {e}")
            return results[:top_k]

    @staticmethod
    def _parse_scores(response: str, expected_count: int) -> List[float]:
        """解析 LLM 返回的分数数组。"""
        import re
        import json

        # 尝试提取 JSON 数组
        match = re.search(r'\[[\d\s,\.]+\]', response)
        if match:
            try:
                scores = json.loads(match.group())
                if len(scores) == expected_count:
                    return [float(s) for s in scores]
            except (json.JSONDecodeError, ValueError):
                pass

        # 降级：均匀分数
        return [5.0] * expected_count
