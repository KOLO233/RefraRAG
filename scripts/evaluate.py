"""评估脚本。

用法：
    python scripts/evaluate.py                        # 全量评估
    python scripts/evaluate.py --max-cases 5          # 评估前5条
    python scripts/evaluate.py --by-level              # 按级别分别评估
    python scripts/evaluate.py --test-set my_set.json  # 指定测试集
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.core.trace import TraceCollector
from src.query_classifier.classifier import create_classifier
from src.generation.pipeline import RAGPipeline
from src.generation.document_grader import DocumentGrader
from src.generation.query_rewriter import QueryRewriter
from src.generation.response_generator import ResponseGenerator
from src.libs.llm_service import LLMService
from src.libs.embedding_service import EmbeddingService
from src.retrieval.milvus_store import MilvusStore
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_search import HybridSearch
from src.ingestion.graph_builder.graph_store import GraphStore
from src.evaluation.evaluator import Evaluator
from src.evaluation.test_set import TestSetManager


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="RefraRAG 评估")
    parser.add_argument("--test-set", default="golden_test_set.json", help="测试集文件名")
    parser.add_argument("--max-cases", type=int, default=0, help="最大评估数（0=全部）")
    parser.add_argument("--by-level", action="store_true", help="按级别分别评估")
    parser.add_argument("--output", "-o", default="", help="结果输出文件")
    args = parser.parse_args()

    settings = load_settings()
    print(f"Settings: LLM={settings.llm.provider}/{settings.llm.model}")
    print(f"Embedding: {settings.embedding.model}")
    print(f"Test set: {args.test_set}")
    print("=" * 60)

    # 初始化所有组件（和 query.py 相同）
    trace_collector = TraceCollector()
    classifier = create_classifier(settings.query_classifier)

    llm_service = LLMService.from_settings(settings)
    grader = DocumentGrader(llm_service)
    rewriter = QueryRewriter(llm_service)

    embedding = EmbeddingService(
        model_name=settings.embedding.model,
        device=settings.embedding.device,
        dimensions=settings.embedding.dimensions,
        api_key=settings.embedding.api_key,
        api_base_url=settings.embedding.api_base_url,
    )
    store = MilvusStore(
        host=settings.vector_store.host,
        port=settings.vector_store.port,
        collection=settings.vector_store.collection,
    )
    dense_retriever = DenseRetriever(embedding, store)
    sparse_retriever = SparseRetriever(embedding, store)

    graph_store = GraphStore(persist_path="data/knowledge_graph.json")
    graph_retriever = GraphRetriever(graph_store) if settings.graph.enabled else None

    reranker = Reranker(llm_service=llm_service)

    hybrid_search = HybridSearch(
        settings=settings,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
    )

    generator = ResponseGenerator(llm_service, hybrid_search)

    pipeline = RAGPipeline(
        settings=settings,
        classifier=classifier,
        hybrid_search=hybrid_search,
        grader=grader,
        rewriter=rewriter,
        generator=generator,
        trace_collector=trace_collector,
    )

    # 运行评估
    evaluator = Evaluator(pipeline=pipeline, llm_service=llm_service)

    if args.by_level:
        results = evaluator.evaluate_by_level(args.test_set)
        print(f"\n{'='*60}")
        print("按级别评估结果：")
        for level, result in results.items():
            print(f"\n  {level}:")
            print(f"    用例数: {result.total_cases}")
            print(f"    分类准确率: {result.classification_accuracy:.2%}")
            print(f"    Hit Rate@5: {result.hit_rate:.2%}")
            print(f"    MRR: {result.mrr:.2%}")
            print(f"    Faithfulness: {result.faithfulness:.2%}")
            print(f"    Answer Relevance: {result.answer_relevance:.2%}")
            print(f"    Context Recall: {result.context_recall:.2%}")
            print(f"    Context Precision: {result.context_precision:.2%}")

        if args.output:
            output_data = {}
            for level, result in results.items():
                output_data[level] = result.to_dict()
            Path(args.output).write_text(
                json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\n结果已保存到: {args.output}")
    else:
        result = evaluator.evaluate(args.test_set, max_cases=args.max_cases)
        print(f"\n{'='*60}")
        print(f"评估结果 ({result.total_cases} 条用例):")
        print(f"  分类准确率:     {result.classification_accuracy:.2%}")
        print(f"  Hit Rate@5:     {result.hit_rate:.2%}")
        print(f"  MRR:            {result.mrr:.2%}")
        print(f"  Faithfulness:   {result.faithfulness:.2%}")
        print(f"  Answer Relevance: {result.answer_relevance:.2%}")
        print(f"  Context Recall: {result.context_recall:.2%}")
        print(f"  Context Precision: {result.context_precision:.2%}")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
