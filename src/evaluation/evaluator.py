"""评估编排器。

编排完整的评估流程：加载测试集 → 运行 Pipeline → 计算指标 → 输出报告。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from src.core.types import EvalTestCase, EvaluationResult
from src.evaluation.metrics import (
    hit_rate_at_k,
    mrr,
    classification_accuracy,
    faithfulness_llm,
    relevancy_llm,
    context_recall,
    context_precision,
)
from src.evaluation.test_set import TestSetManager

logger = logging.getLogger(__name__)


class Evaluator:
    """评估编排器。

    Example:
        >>> evaluator = Evaluator(pipeline, llm_service)
        >>> result = evaluator.evaluate("golden_test_set.json")
        >>> print(f"Hit Rate: {result.hit_rate:.2%}")
        >>> print(f"MRR: {result.mrr:.2%}")
    """

    def __init__(
        self,
        pipeline=None,
        llm_service=None,
        test_set_manager: Optional[TestSetManager] = None,
    ):
        self._pipeline = pipeline
        self._llm = llm_service
        self._test_set_mgr = test_set_manager or TestSetManager()

    def evaluate(
        self,
        test_set_file: str = "golden_test_set.json",
        max_cases: int = 0,
    ) -> EvaluationResult:
        """运行完整评估。

        Args:
            test_set_file: 测试集文件名
            max_cases: 最大评估用例数（0=全部）

        Returns:
            EvaluationResult 包含所有指标
        """
        test_cases = self._test_set_mgr.load(test_set_file)
        if not test_cases:
            logger.error(f"No test cases loaded from {test_set_file}")
            return EvaluationResult()

        if max_cases > 0:
            test_cases = test_cases[:max_cases]

        logger.info(f"Evaluating {len(test_cases)} test cases...")

        predicted_levels = []
        expected_levels = []
        hit_rates = []
        mrr_scores = []
        faithfulness_scores = []
        relevancy_scores = []
        context_recall_scores = []
        context_precision_scores = []
        details = []

        for i, tc in enumerate(test_cases):
            logger.info(f"  [{i+1}/{len(test_cases)}] {tc.question[:50]}...")

            try:
                # 运行 Pipeline
                t0 = time.monotonic()
                response = asyncio.run(self._pipeline.run(tc.question))
                elapsed = (time.monotonic() - t0) * 1000

                # 记录分类结果
                predicted_levels.append(response.query_level)
                expected_levels.append(tc.expected_level)

                # 检索指标
                retrieved_ids = [
                    c.source for c in response.citations
                ]
                if tc.ground_truth_chunks:
                    hr = hit_rate_at_k(retrieved_ids, tc.ground_truth_chunks, k=5)
                    mrr_score = mrr(retrieved_ids, tc.ground_truth_chunks, k=10)
                    hit_rates.append(hr)
                    mrr_scores.append(mrr_score)
                else:
                    hr = None
                    mrr_score = None

                # 生成指标（与 LightRAG/RAGAS 对齐）
                context = "\n".join([c.text_snippet for c in response.citations])
                faith = faithfulness_llm(response.answer, context, self._llm)
                relev = relevancy_llm(tc.question, response.answer, self._llm)
                ctx_recall = context_recall(context, tc.expected_answer, self._llm)
                ctx_precision = context_precision(context, tc.question, tc.expected_answer, self._llm)
                faithfulness_scores.append(faith)
                relevancy_scores.append(relev)
                context_recall_scores.append(ctx_recall)
                context_precision_scores.append(ctx_precision)

                detail = {
                    "question": tc.question,
                    "expected_level": tc.expected_level,
                    "predicted_level": response.query_level,
                    "level_correct": response.query_level == tc.expected_level,
                    "hit_rate": hr,
                    "mrr": mrr_score,
                    "faithfulness": faith,
                    "answer_relevance": relev,
                    "context_recall": ctx_recall,
                    "context_precision": ctx_precision,
                    "elapsed_ms": elapsed,
                    "answer_preview": response.answer[:200],
                }
                details.append(detail)

            except Exception as e:
                logger.error(f"  Failed: {e}")
                details.append({
                    "question": tc.question,
                    "expected_level": tc.expected_level,
                    "error": str(e),
                })

        # 汇总指标（与 LightRAG/RAGAS 对齐的 6 个指标）
        result = EvaluationResult(
            hit_rate=_avg(hit_rates) if hit_rates else 0.0,
            mrr=_avg(mrr_scores) if mrr_scores else 0.0,
            faithfulness=_avg(faithfulness_scores) if faithfulness_scores else 0.0,
            answer_relevance=_avg(relevancy_scores) if relevancy_scores else 0.0,
            context_recall=_avg(context_recall_scores) if context_recall_scores else 0.0,
            context_precision=_avg(context_precision_scores) if context_precision_scores else 0.0,
            classification_accuracy=classification_accuracy(predicted_levels, expected_levels),
            total_cases=len(test_cases),
            details=details,
        )

        logger.info(f"Evaluation complete:")
        logger.info(f"  Classification Accuracy: {result.classification_accuracy:.2%}")
        logger.info(f"  Hit Rate@5: {result.hit_rate:.2%}")
        logger.info(f"  MRR: {result.mrr:.2%}")
        logger.info(f"  Faithfulness: {result.faithfulness:.2%}")
        logger.info(f"  Answer Relevance: {result.answer_relevance:.2%}")
        logger.info(f"  Context Recall: {result.context_recall:.2%}")
        logger.info(f"  Context Precision: {result.context_precision:.2%}")

        return result

    def evaluate_by_level(
        self,
        test_set_file: str = "golden_test_set.json",
    ) -> Dict[str, EvaluationResult]:
        """按级别分别评估。"""
        test_cases = self._test_set_mgr.load(test_set_file)
        if not test_cases:
            return {}

        results = {}
        for level in ["L1", "L2", "L3", "L4"]:
            level_cases = self._test_set_mgr.filter_by_level(test_cases, level)
            if not level_cases:
                continue

            # 临时保存级别测试集
            level_file = f"_temp_{level}.json"
            self._test_set_mgr.save(level_file, level_cases)

            result = self.evaluate(level_file)
            results[level] = result

            # 清理临时文件
            temp_path = self._test_set_mgr._data_dir / level_file
            if temp_path.exists():
                temp_path.unlink()

        return results


def _avg(values: List[float]) -> float:
    """计算平均值。"""
    return sum(values) / len(values) if values else 0.0
