"""Graph Retriever — 知识图谱检索器。

参考 LightRAG 的 kg_query 思路：
1. 从查询中提取关键词/实体
2. 在图中定位相关实体
3. 多跳遍历获取关联实体和关系
4. 将子图转化为文本上下文

L3 使用多跳遍历（因果链），L4 使用单跳（事实锚点）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from src.core.types import RetrievalResult
from src.ingestion.graph_builder.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphRetriever:
    """知识图谱检索器。

    流程：
    1. 实体定位：从查询中提取关键词，在图中搜索匹配实体
    2. 子图遍历：从匹配实体出发，多跳遍历获取关联信息
    3. 上下文构建：将子图信息转化为文本

    Example:
        >>> retriever = GraphRetriever(graph_store)
        >>> results = await retriever.retrieve("为什么会出现梯度消失？", hops=2)
    """

    def __init__(self, graph_store: GraphStore):
        self._store = graph_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        hops: int = 2,
    ) -> List[RetrievalResult]:
        """执行图检索。

        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            hops: 遍历跳数（L3 用 2，L4 用 1）

        Returns:
            RetrievalResult 列表，每个结果包含一段子图上下文
        """
        # Step 1: 提取查询关键词
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # Step 2: 在图中搜索匹配实体
        matched_entities = []
        for kw in keywords:
            found = self._store.search_entities(kw)
            matched_entities.extend(found)

        if not matched_entities:
            logger.debug(f"Graph: no entities matched for keywords {keywords}")
            return []

        # Step 3: 从匹配实体出发，多跳遍历
        all_subgraph_contexts = []
        seen_entities = set()

        for entity in matched_entities[:5]:  # 最多从 5 个实体出发
            entity_name = entity["name"]
            if entity_name in seen_entities:
                continue
            seen_entities.add(entity_name)

            neighbors = self._store.get_neighbors(
                entity_name, hops=hops, max_neighbors=10
            )

            # Step 4: 将子图转化为文本上下文
            context_text = self._subgraph_to_text(entity_name, neighbors)
            if context_text:
                all_subgraph_contexts.append({
                    "center": entity_name,
                    "text": context_text,
                    "entity_count": len(neighbors.get("entities", [])),
                    "relation_count": len(neighbors.get("relations", [])),
                })

        # 按关联实体数量排序（关联越多越相关）
        all_subgraph_contexts.sort(
            key=lambda x: x["entity_count"] + x["relation_count"],
            reverse=True,
        )

        # 转换为 RetrievalResult
        results = []
        for i, ctx in enumerate(all_subgraph_contexts[:top_k]):
            results.append(RetrievalResult(
                chunk_id=f"graph::{ctx['center']}",
                score=1.0 / (i + 1),  # 简单排名分数
                text=ctx["text"],
                metadata={
                    "source_path": "knowledge_graph",
                    "filename": "knowledge_graph",
                    "center_entity": ctx["center"],
                    "entity_count": ctx["entity_count"],
                    "relation_count": ctx["relation_count"],
                    "hops": hops,
                },
                retrieval_source="graph",
            ))

        logger.debug(
            f"Graph retrieval: keywords={keywords}, "
            f"matched={len(matched_entities)}, results={len(results)}"
        )
        return results

    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词用于实体匹配。"""
        import jieba
        # 用 jieba 分词，过滤停用词和单字
        words = jieba.cut(query)
        stop_words = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "个",
                       "什么", "怎么", "为什么", "如何", "可以", "会", "将", "被",
                       "如果", "出现", "使用", "比较", "哪些"}
        keywords = []
        for w in words:
            w = w.strip()
            if len(w) >= 2 and w not in stop_words:
                keywords.append(w)
        # 同时保留原始查询中较长的连续中文片段
        import re
        for match in re.finditer(r'[一-鿿]{2,}', query):
            word = match.group()
            if word not in keywords and word not in stop_words:
                keywords.append(word)
        return list(set(keywords))

    def _subgraph_to_text(self, center: str, neighbors: Dict) -> str:
        """将子图信息转化为可读文本。

        格式参考 LightRAG 的实体/关系上下文格式。
        """
        entities = neighbors.get("entities", [])
        relations = neighbors.get("relations", [])

        if not entities and not relations:
            return ""

        lines = []
        lines.append(f"## 关于「{center}」的知识图谱信息")

        # 实体信息
        if entities:
            lines.append("\n### 相关实体")
            for e in entities:
                hop_str = f" (第{e['hop']}跳)" if e.get("hop", 1) > 1 else ""
                desc = f": {e['description']}" if e.get("description") else ""
                lines.append(f"- **{e['name']}** [{e.get('entity_type', '')}]{hop_str}{desc}")

        # 关系信息
        if relations:
            lines.append("\n### 关系链")
            for r in relations:
                hop_str = f" (第{r['hop']}跳)" if r.get("hop", 1) > 1 else ""
                desc = f": {r['description']}" if r.get("description") else ""
                lines.append(f"- **{r['source']}** →[{r.get('relation_type', '')}]→ **{r['target']}**{hop_str}{desc}")

        return "\n".join(lines)
