"""嵌入服务。

提供文本向量化能力，支持：
- 密集向量（Dense）：使用 BGE-M3 本地模型
- 稀疏向量（Sparse）：手动 BM25 实现，统计持久化

设计参考 SuperMew 的 EmbeddingService，但接口更简洁。
"""

from __future__ import annotations

import json
import logging
import math
import os
# 强制离线模式，避免 sentence-transformers 每次启动都连接 HuggingFace
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bm25_state.json"


class EmbeddingService:
    """文本向量化服务。

    提供密集向量（BGE-M3）和稀疏向量（BM25）两种向量化方式。

    Example:
        >>> service = EmbeddingService(model_name="BAAI/bge-m3")
        >>> dense = service.embed_dense(["什么是机器学习？"])
        >>> print(len(dense[0]))  # 1024
        >>> sparse = service.embed_sparse(["什么是机器学习？"])
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        dimensions: int = 1024,
        state_path: Optional[str | Path] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        self._model_name = model_name
        self._device = device
        self._dimensions = dimensions
        self._embedder = None  # 懒加载
        self._api_key = api_key
        self._api_base_url = api_base_url
        self._use_api = api_key and api_base_url  # 有 API 配置时用 API

        # BM25 参数
        self._state_path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self._lock = threading.Lock()
        self.k1 = 1.5
        self.b = 0.75
        self._vocab: Dict[str, int] = {}
        self._vocab_counter = 0
        self._doc_freq: Counter = Counter()
        self._total_docs = 0
        self._sum_token_len = 0
        self._avg_doc_len = 1.0
        self._load_bm25_state()

    # =========================================================================
    # 密集向量 (Dense)
    # =========================================================================

    def _embed_via_api(self, texts: List[str]) -> List[List[float]]:
        """通过 OpenAI 兼容 API 生成密集向量。"""
        import httpx

        url = f"{self._api_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model_name,
            "input": texts,
        }

        resp = httpx.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # 按 index 排序保证顺序正确
        sorted_results = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_results]

    def _get_embedder(self):
        """懒加载嵌入模型。"""
        if self._embedder is None:
            if self._model_name == "lightweight":
                # 轻量后端：jieba 分词 + TF-IDF，不依赖 PyTorch
                self._embedder = self._create_lightweight_embedder()
                logger.info("Using lightweight embedder (jieba + TF-IDF)")
            else:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    self._embedder = HuggingFaceEmbeddings(
                        model_name=self._model_name,
                        model_kwargs={"device": self._device},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                    logger.info(f"Loaded embedding model: {self._model_name} on {self._device}")
                except Exception as e:
                    logger.error(f"Failed to load model {self._model_name}: {e}")
                    logger.info("Falling back to lightweight embedder")
                    self._embedder = self._create_lightweight_embedder()
        return self._embedder

    @staticmethod
    def _create_lightweight_embedder():
        """创建轻量嵌入器（jieba + TF-IDF），不依赖 PyTorch。"""
        import math
        import numpy as np

        try:
            import jieba
        except ImportError:
            raise ImportError("jieba is required for lightweight embedder. Install: pip install jieba")

        class LightweightEmbedder:
            """jieba 分词 + TF-IDF 向量化。不依赖 PyTorch，瞬间可用。"""

            def __init__(self):
                self._vocab = {}       # token -> index
                self._idf = {}         # token -> idf score
                self._doc_count = 0
                self._fitted = False

            def _tokenize(self, text: str) -> list:
                return [w for w in jieba.cut(text) if len(w.strip()) > 0]

            def _fit(self, texts: list):
                """基于语料库构建 IDF。"""
                if self._fitted:
                    return
                df = Counter()
                self._doc_count = len(texts)
                for text in texts:
                    tokens = set(self._tokenize(text))
                    for t in tokens:
                        df[t] += 1
                # 构建词表
                self._vocab = {t: i for i, t in enumerate(df.keys())}
                # 计算 IDF
                for t, freq in df.items():
                    self._idf[t] = math.log((self._doc_count + 1) / (freq + 1)) + 1
                self._fitted = True

            def _text_to_vec(self, text: str, dim: int = 512) -> list:
                """将文本转成 TF-IDF 向量。"""
                tokens = self._tokenize(text)
                if not tokens:
                    return [0.0] * dim

                tf = Counter(tokens)
                vec = {}
                for t, count in tf.items():
                    if t in self._idf:
                        idx = self._vocab.get(t, hash(t) % dim)
                        vec[idx % dim] = vec.get(idx % dim, 0) + count * self._idf[t]

                # 转为固定维度向量并归一化
                result = [0.0] * dim
                for idx, val in vec.items():
                    result[idx] = val

                # L2 归一化
                norm = math.sqrt(sum(v * v for v in result)) or 1.0
                result = [v / norm for v in result]
                return result

            def embed_documents(self, texts: list) -> list:
                self._fit(texts)
                return [self._text_to_vec(t) for t in texts]

            def embed_query(self, query: str) -> list:
                if not self._fitted:
                    self._fit([query])
                return self._text_to_vec(query)

        return LightweightEmbedder()

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """生成密集向量。"""
        if not texts:
            return []
        if self._use_api:
            return self._embed_via_api(texts)
        embedder = self._get_embedder()
        return embedder.embed_documents(texts)

    def embed_dense_query(self, query: str) -> List[float]:
        """为查询生成密集向量。"""
        if self._use_api:
            return self._embed_via_api([query])[0]
        embedder = self._get_embedder()
        return embedder.embed_query(query)

    @property
    def dense_dim(self) -> int:
        return self._dimensions

    # =========================================================================
    # 稀疏向量 (Sparse / BM25)
    # =========================================================================

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中英文混合分词。

        中文按单字切分，英文按空格切分。
        """
        tokens = []
        # 英文单词
        english_words = re.findall(r'[a-zA-Z]+(?:\.\w+)*', text)
        tokens.extend(english_words)
        # 中文字符
        chinese_chars = re.findall(r'[一-鿿]', text)
        tokens.extend(chinese_chars)
        # 数字
        numbers = re.findall(r'\d+', text)
        tokens.extend(numbers)
        return tokens

    def _load_bm25_state(self) -> None:
        """从文件加载 BM25 统计状态。"""
        if not self._state_path.is_file():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if raw.get("version") != 1:
            return
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        self._total_docs = int(raw.get("total_docs", 0))
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        if self._vocab:
            self._vocab_counter = max(self._vocab.values()) + 1
        self._recompute_avg_len()

    def _persist_bm25_state(self) -> None:
        """持久化 BM25 统计到文件。"""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "total_docs": self._total_docs,
            "sum_token_len": self._sum_token_len,
            "vocab": self._vocab,
            "doc_freq": dict(self._doc_freq),
        }
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._state_path)

    def _recompute_avg_len(self) -> None:
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def bm25_increment_add(self, texts: List[str]) -> None:
        """增量添加文档到 BM25 统计。"""
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._total_docs += 1
                self._sum_token_len += doc_len
                seen = set()
                for token in tokens:
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    if token not in seen:
                        self._doc_freq[token] += 1
                        seen.add(token)
            self._recompute_avg_len()
            self._persist_bm25_state()

    def bm25_increment_remove(self, texts: List[str]) -> None:
        """增量从 BM25 统计中移除文档。"""
        if not texts:
            return
        with self._lock:
            for text in texts:
                tokens = self.tokenize(text)
                doc_len = len(tokens)
                self._total_docs = max(0, self._total_docs - 1)
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                seen = set()
                for token in tokens:
                    if token not in seen:
                        self._doc_freq[token] = max(0, self._doc_freq.get(token, 0) - 1)
                        seen.add(token)
            self._recompute_avg_len()
            self._persist_bm25_state()

    def compute_bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """计算单个文档的 BM25 分数。"""
        doc_len = len(doc_tokens)
        tf_map: Dict[str, int] = Counter(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            df = self._doc_freq.get(qt, 0)
            if df == 0:
                continue
            idf = math.log(
                (self._total_docs - df + 0.5) / (df + 0.5) + 1.0
            )
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
            )
            score += idf * tf_norm
        return score

    def embed_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        """为文本生成 BM25 稀疏向量。

        Returns:
            每个文本的稀疏向量，格式 {token_index: bm25_score}
        """
        results = []
        for text in texts:
            tokens = self.tokenize(text)
            query_tokens = list(set(tokens))
            scores: Dict[str, float] = {}

            for qt in query_tokens:
                tf = tokens.count(qt)
                df = self._doc_freq.get(qt, 0)
                if df == 0 or self._total_docs == 0:
                    continue
                idf = math.log(
                    (self._total_docs - df + 0.5) / (df + 0.5) + 1.0
                )
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * len(tokens) / self._avg_doc_len)
                )
                idx = self._vocab.get(qt)
                if idx is not None:
                    scores[str(idx)] = idf * tf_norm

            results.append(scores)
        return results

    def embed_all(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """同时生成密集和稀疏向量。"""
        dense = self.embed_dense(texts)
        sparse = self.embed_sparse(texts)
        return dense, sparse
