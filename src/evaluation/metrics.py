"""评估指标实现。

支持的指标：
- Hit Rate@K：Top-K 中是否包含正确文档
- MRR (Mean Reciprocal Rank)：正确文档的平均倒数排名
- Faithfulness：回答是否忠于检索内容
- Relevancy：回答与问题的相关性
- Classification Accuracy：查询分类准确率
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def hit_rate_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int = 5,
) -> float:
    """Hit Rate@K：Top-K 检索结果中是否命中正确文档。

    Args:
        retrieved_ids: 检索结果的 chunk_id 列表（按相关性排序）
        ground_truth_ids: 标注的正确 chunk_id 列表
        k: 截断位置

    Returns:
        1.0 如果命中，0.0 如果未命中
    """
    if not ground_truth_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gt = set(ground_truth_ids)
    return 1.0 if top_k & gt else 0.0


def mrr(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int = 10,
) -> float:
    """MRR (Mean Reciprocal Rank)：正确文档的倒数排名。

    Args:
        retrieved_ids: 检索结果的 chunk_id 列表
        ground_truth_ids: 标注的正确 chunk_id 列表
        k: 截断位置

    Returns:
        1/rank 的值（rank 为第一个正确结果的位置，1-indexed）
    """
    if not ground_truth_ids:
        return 0.0
    gt = set(ground_truth_ids)
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gt:
            return 1.0 / (i + 1)
    return 0.0


def classification_accuracy(
    predicted_levels: List[str],
    expected_levels: List[str],
) -> float:
    """查询分类准确率。

    Args:
        predicted_levels: 预测的级别列表 (L1/L2/L3/L4)
        expected_levels: 标注的级别列表

    Returns:
        准确率 (0.0 ~ 1.0)
    """
    if not predicted_levels or not expected_levels:
        return 0.0
    correct = sum(1 for p, e in zip(predicted_levels, expected_levels) if p == e)
    return correct / len(expected_levels)


def faithfulness_llm(
    answer: str,
    context: str,
    llm_service=None,
) -> float:
    """Faithfulness：回答是否忠于检索内容（需要 LLM 评估）。

    Args:
        answer: 生成的回答
        context: 检索到的上下文
        llm_service: LLM 服务（可选，无则用规则评估）

    Returns:
        忠实度分数 (0.0 ~ 1.0)
    """
    if not answer or not context:
        return 0.0

    if llm_service is None:
        return _rule_based_faithfulness(answer, context)

    prompt = f"""请评估以下回答是否忠于参考资料（不编造信息）。

参考资料：
{context[:2000]}

回答：
{answer[:1000]}

评分标准：
- 1.0：回答完全基于参考资料，无编造
- 0.7-0.9：回答大部分基于参考资料，有少量合理推断
- 0.4-0.6：回答部分基于参考资料，有一定编造
- 0.1-0.3：回答大部分编造，少量引用参考资料
- 0.0：回答完全编造

请仅输出一个 0.0 到 1.0 之间的数字。"""

    try:
        import asyncio
        response = asyncio.run(llm_service.ainvoke(prompt))
        # 提取数字
        import re
        match = re.search(r'(0\.\d+|1\.0|0|1)', response)
        if match:
            return float(match.group())
    except Exception as e:
        logger.error(f"LLM faithfulness evaluation failed: {e}")

    return _rule_based_faithfulness(answer, context)


def relevancy_llm(
    question: str,
    answer: str,
    llm_service=None,
) -> float:
    """Relevancy：回答与问题的相关性（需要 LLM 评估）。

    Args:
        question: 用户问题
        answer: 生成的回答
        llm_service: LLM 服务（可选）

    Returns:
        相关性分数 (0.0 ~ 1.0)
    """
    if not question or not answer:
        return 0.0

    if llm_service is None:
        return _rule_based_relevancy(question, answer)

    prompt = f"""请评估以下回答与问题的相关性。

问题：{question}

回答：{answer[:1000]}

评分标准：
- 1.0：回答完全切题，准确回答了问题
- 0.7-0.9：回答基本切题，涵盖了问题的核心
- 0.4-0.6：回答部分切题，遗漏了关键信息
- 0.1-0.3：回答偏离主题
- 0.0：回答完全不相关

请仅输出一个 0.0 到 1.0 之间的数字。"""

    try:
        import asyncio
        response = asyncio.run(llm_service.ainvoke(prompt))
        import re
        match = re.search(r'(0\.\d+|1\.0|0|1)', response)
        if match:
            return float(match.group())
    except Exception as e:
        logger.error(f"LLM relevancy evaluation failed: {e}")

    return _rule_based_relevancy(question, answer)


def context_recall(
    context: str,
    reference_answer: str,
    llm_service=None,
) -> float:
    """Context Recall：检索到的上下文是否覆盖了参考答案中的关键信息。

    与 LightRAG/RAGAS 的 context_recall 对齐。
    衡量检索系统的完整性——是否找到了所有相关信息。

    Args:
        context: 检索到的上下文
        reference_answer: 标注的参考答案
        llm_service: LLM 服务（可选）

    Returns:
        召回率分数 (0.0 ~ 1.0)
    """
    if not context or not reference_answer:
        return 0.0

    if llm_service is None:
        return _rule_based_context_recall(context, reference_answer)

    prompt = f"""请评估检索到的上下文是否覆盖了参考答案中的关键信息。

参考答案：
{reference_answer[:1000]}

检索到的上下文：
{context[:2000]}

评分标准：
- 1.0：上下文完全覆盖了参考答案的所有关键信息
- 0.7-0.9：上下文覆盖了大部分关键信息
- 0.4-0.6：上下文覆盖了部分关键信息
- 0.1-0.3：上下文仅覆盖了少量关键信息
- 0.0：上下文未覆盖任何关键信息

请仅输出一个 0.0 到 1.0 之间的数字。"""

    try:
        import asyncio
        response = asyncio.run(llm_service.ainvoke(prompt))
        import re
        match = re.search(r'(0\.\d+|1\.0|0|1)', response)
        if match:
            return float(match.group())
    except Exception as e:
        logger.error(f"LLM context_recall evaluation failed: {e}")

    return _rule_based_context_recall(context, reference_answer)


def context_precision(
    context: str,
    question: str,
    reference_answer: str,
    llm_service=None,
) -> float:
    """Context Precision：检索到的上下文是否干净、无噪声。

    与 LightRAG/RAGAS 的 context_precision 对齐。
    衡量检索系统的精确性——检索到的内容是否都相关。

    Args:
        context: 检索到的上下文
        question: 用户问题
        reference_answer: 标注的参考答案
        llm_service: LLM 服务（可选）

    Returns:
        精确率分数 (0.0 ~ 1.0)
    """
    if not context:
        return 0.0

    if llm_service is None:
        return _rule_based_context_precision(context, question, reference_answer)

    prompt = f"""请评估检索到的上下文是否干净（无噪声）且与问题相关。

问题：{question}

参考答案：{reference_answer[:500]}

检索到的上下文：
{context[:2000]}

评分标准：
- 1.0：上下文全部与问题相关，无噪声
- 0.7-0.9：上下文大部分相关，少量噪声
- 0.4-0.6：上下文部分相关，有一定噪声
- 0.1-0.3：上下文大部分是噪声
- 0.0：上下文完全不相关

请仅输出一个 0.0 到 1.0 之间的数字。"""

    try:
        import asyncio
        response = asyncio.run(llm_service.ainvoke(prompt))
        import re
        match = re.search(r'(0\.\d+|1\.0|0|1)', response)
        if match:
            return float(match.group())
    except Exception as e:
        logger.error(f"LLM context_precision evaluation failed: {e}")

    return _rule_based_context_precision(context, question, reference_answer)


# ===========================================================================
# 规则基线评估（不需要 LLM）
# ===========================================================================

def _rule_based_faithfulness(answer: str, context: str) -> float:
    """规则基线 Faithfulness：基于关键词重叠度。"""
    import jieba

    ans_tokens = set(jieba.cut(answer))
    ctx_tokens = set(jieba.cut(context))

    # 去除停用词
    stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "个", "与", "对", "中", "上", "下"}
    ans_tokens -= stop
    ctx_tokens -= stop

    if not ans_tokens:
        return 0.0

    overlap = ans_tokens & ctx_tokens
    return min(len(overlap) / len(ans_tokens), 1.0)


def _rule_based_relevancy(question: str, answer: str) -> float:
    """规则基线 Relevancy：基于问题关键词在回答中的覆盖率。"""
    import jieba
    import re

    # 提取问题关键词
    q_tokens = set(jieba.cut(question))
    stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "什么", "怎么", "为什么", "如何", "哪些"}
    q_tokens -= stop
    q_tokens = {t for t in q_tokens if len(t) >= 2}

    if not q_tokens:
        return 0.0

    a_lower = answer.lower()
    matched = sum(1 for t in q_tokens if t in a_lower)
    return min(matched / len(q_tokens), 1.0)


def _rule_based_context_recall(context: str, reference_answer: str) -> float:
    """规则基线 Context Recall：参考答案的关键信息在上下文中的覆盖率。"""
    import jieba

    ref_tokens = set(jieba.cut(reference_answer))
    ctx_tokens = set(jieba.cut(context))

    stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "个", "与", "对", "中"}
    ref_tokens -= stop
    ref_tokens = {t for t in ref_tokens if len(t) >= 2}
    ctx_tokens -= stop

    if not ref_tokens:
        return 0.0

    overlap = ref_tokens & ctx_tokens
    return min(len(overlap) / len(ref_tokens), 1.0)


def _rule_based_context_precision(context: str, question: str, reference_answer: str) -> float:
    """规则基线 Context Precision：上下文中与问题/答案相关内容的比例。"""
    import jieba

    # 合并问题和参考答案的关键词作为"相关"标准
    relevant_tokens = set(jieba.cut(question + reference_answer))
    stop = {"的", "了", "是", "在", "和", "有", "为", "这", "那", "个", "什么", "怎么", "为什么"}
    relevant_tokens -= stop
    relevant_tokens = {t for t in relevant_tokens if len(t) >= 2}

    ctx_tokens = set(jieba.cut(context))
    ctx_tokens -= stop
    ctx_tokens = {t for t in ctx_tokens if len(t) >= 2}

    if not ctx_tokens:
        return 0.0

    overlap = ctx_tokens & relevant_tokens
    return min(len(overlap) / len(ctx_tokens), 1.0)
