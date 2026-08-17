# ============================================================
# src/retrieval/embedding.py - Dense vector retrieval via local embeddings
# ============================================================

import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.utils.config_loader import config


class EmbeddingRetriever:
    """
    Dense retrieval using local embedding models.

    Primary:   Qwen3-Embedding-0.6B  (0.6B params, 4096-dim)
    Fallback:  all-MiniLM-L6-v2       (22M params, 384-dim)

    Model paths are configured in config/.env:
      EMBEDDING_MODEL_PATH = D:/models/Qwen/Qwen3-Embedding-0.6B
      EMBEDDING_MODEL_PATH_LIGHT = D:/models/all-MiniLM-L6-v2
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embeddings: Optional[np.ndarray] = None
        self._docs: List[Dict[str, Any]] = []
        self._model = None

        self._init_model()

    def _init_model(self) -> None:
        """Load the embedding model from local path."""
        # Determine model path from .env
        if "qwen" in self.model_name.lower() or "0.6b" in self.model_name.lower():
            model_path = config.get_api_key("EMBEDDING_MODEL_PATH")
            fallback_name = "all-MiniLM-L6-v2"
            fallback_path = config.get_api_key("EMBEDDING_MODEL_PATH_LIGHT")
        else:
            # Lightweight model
            model_path = config.get_api_key("EMBEDDING_MODEL_PATH_LIGHT")
            fallback_name = "qwen3-embedding-0.6b"
            fallback_path = config.get_api_key("EMBEDDING_MODEL_PATH")

        if not model_path or not os.path.exists(str(model_path)):
            print(f"[WARN] Embedding model not found at: {model_path}")
            if fallback_path and os.path.exists(str(fallback_path)):
                print(f"       Falling back to: {fallback_path}")
                model_path = fallback_path
                self.model_name = fallback_name
            else:
                print(f"       Falling back to simple TF-IDF embedding")
                self._model = None
                return

        try:
            from sentence_transformers import SentenceTransformer
            device = config.get("retrieval", "embedding", "device", default="cpu")
            self._model = SentenceTransformer(model_path, device=device)
            print(f"[INFO] Embedding model loaded: {self.model_name} ({model_path})")
        except ImportError:
            print("[WARN] sentence-transformers not installed. Install with:")
            print("       pip install sentence-transformers")
            print("       Falling back to simple TF-IDF embedding")
            self._model = None
        except Exception as e:
            print(f"[WARN] Failed to load embedding model: {e}")
            print(f"       Falling back to simple TF-IDF embedding")
            self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        if texts is None or len(texts) == 0:
            return np.array([])

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.array(embeddings)
        else:
            return self._tfidf_fallback(texts)

    def _tfidf_fallback(self, texts: List[str]) -> np.ndarray:
        """Simple TF-IDF fallback when no embedding model is available."""
        from collections import Counter
        import math

        tokenized = []
        for text in texts:
            tokens = text.lower().replace(",", " ").replace(".", " ").split()
            tokenized.append(tokens)

        N = len(texts)
        df = Counter()
        for tokens in tokenized:
            for t in set(tokens):
                df[t] += 1

        dim = min(1024, max(100, len(df) // 2))
        vocab = [t for t, _ in df.most_common(dim)]
        vocab_idx = {t: i for i, t in enumerate(vocab)}

        vectors = np.zeros((N, dim))
        for i, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            for t, f in tf.items():
                if t in vocab_idx:
                    idf = math.log((N + 1) / (df[t] + 1)) + 1
                    vectors[i, vocab_idx[t]] = f * idf
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm

        return vectors

    def fit(self, documents: List[Dict[str, Any]]) -> "EmbeddingRetriever":
        """Index a list of paper dicts."""
        self._docs = documents
        texts = []
        for doc in documents:
            title = doc.get("title") or ""
            abstract = doc.get("abstract") or ""
            texts.append((title + " " + abstract)[:2000])
        self._embeddings = self.encode(texts)
        return self

    def search(self, query: str, top_k: int = 50) -> List[Tuple[Dict[str, Any], float]]:
        """Search by embedding cosine similarity."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query_vec = self.encode([query])
        if query_vec.shape[0] == 0:
            return []

        # Cosine similarity (embeddings are already normalized)
        similarities = (query_vec @ self._embeddings.T).flatten()

        indices = np.argsort(-similarities)[:top_k]
        results = []
        for idx in indices:
            if similarities[idx] > 0:
                results.append((self._docs[idx], float(similarities[idx])))

        return results

    def get_top_n(self, query: str, n: int = 50) -> List[Dict[str, Any]]:
        """Convenience: get top-N papers."""
        return [doc for doc, _ in self.search(query, top_k=n)]


def build_embedding_index(papers: List[Dict[str, Any]]) -> EmbeddingRetriever:
    """Factory: create and fit an EmbeddingRetriever from paper list."""
    model_name = config.get("retrieval", "embedding", "model", default="all-MiniLM-L6-v2")
    retriever = EmbeddingRetriever(model_name=model_name)
    retriever.fit(papers)
    return retriever
