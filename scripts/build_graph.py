"""知识图谱构建脚本。

用法：
    python scripts/build_graph.py --input data/documents/
    python scripts/build_graph.py --input data/documents/AI与机器学习基础教程.md
    python scripts/build_graph.py --stats   # 查看图统计
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import load_settings
from src.ingestion.loaders.markdown_loader import MarkdownLoader
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.graph_builder.graph_store import GraphStore
from src.ingestion.graph_builder.graph_builder import GraphBuilder
from src.libs.llm_service import LLMService


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="RefraRAG 知识图谱构建")
    parser.add_argument("--input", "-i", help="输入文件或目录")
    parser.add_argument("--stats", action="store_true", help="显示图统计信息")
    parser.add_argument("--query", "-q", help="查询图谱")
    args = parser.parse_args()

    settings = load_settings()
    graph_store = GraphStore(persist_path="data/knowledge_graph.json")

    if args.stats:
        stats = graph_store.stats()
        print(f"实体数量: {stats['entity_count']}")
        print(f"关系数量: {stats['relation_count']}")
        return

    if args.query:
        results = graph_store.search_entities(args.query)
        if results:
            print(f"找到 {len(results)} 个匹配实体:")
            for r in results:
                print(f"  - {r['name']} [{r.get('entity_type', '')}]: {r.get('description', '')[:80]}")

                neighbors = graph_store.get_neighbors(r["name"], hops=2)
                for rel in neighbors.get("relations", []):
                    print(f"    → {rel['source']} --[{rel.get('relation_type', '')}]--> {rel['target']}")
        else:
            print(f"未找到匹配 '{args.query}' 的实体")
        return

    if not args.input:
        parser.print_help()
        return

    input_path = Path(args.input)
    loaders = {".pdf": PDFLoader(), ".md": MarkdownLoader(), ".markdown": MarkdownLoader()}

    # 加载文档
    chunks = []
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
    for f in files:
        loader = loaders.get(f.suffix.lower())
        if loader:
            docs = loader.load(f)
            from src.ingestion.chunking.document_chunker import DocumentChunker
            chunker = DocumentChunker(settings.ingestion)
            for doc in docs:
                leaf_chunks, _ = chunker.split_document(doc)
                chunks.extend(leaf_chunks)

    print(f"加载了 {len(chunks)} 个分块")

    # 构建图谱
    llm_service = LLMService.from_settings(settings)
    builder = GraphBuilder(settings.graph, llm_service, graph_store)
    stats = await builder.build_from_chunks(chunks)

    print(f"\n{'='*50}")
    print(f"图谱构建完成:")
    print(f"  处理分块: {stats['processed_chunks']}")
    print(f"  实体数量: {stats['total_entities']}")
    print(f"  关系数量: {stats['total_relations']}")


if __name__ == "__main__":
    asyncio.run(main())
