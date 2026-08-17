# ============================================================
# src/retrieval/bm25.py - BM25 sparse retrieval
# ============================================================

import re
import math
from typing import List, Dict, Any, Tuple
from collections import Counter
from src.utils.config_loader import config


class BM25Retriever:
    """
    BM25 (Okapi BM25) sparse retrieval.
    Used for local re-ranking of API search results.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._avgdl: float = 0.0
        self._df: Counter = Counter()   # Document frequency
        self._N: int = 0

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        if not text:
            return []
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def fit(self, documents: List[Dict[str, Any]]) -> "BM25Retriever":
        """
        Build BM25 index from a list of paper dicts.
        Each doc must have 'title' and optionally 'abstract'.
        """
        self._N = len(documents)
        doc_texts = []
        for doc in documents:
            title = doc.get("title") or ""
            abstract = doc.get("abstract") or ""
            text = title + " " + abstract
            tokens = self.tokenize(text)
            doc_texts.append(tokens)

        # Compute average document length
        self._avgdl = sum(len(t) for t in doc_texts) / max(self._N, 1)

        # Compute document frequencies
        self._df = Counter()
        for tokens in doc_texts:
            unique = set(tokens)
            for token in unique:
                self._df[token] += 1

        # Store tokenized docs for scoring
        self._doc_tokens = doc_texts
        self._docs = documents

        return self

    def score(self, query: str) -> List[Tuple[int, float]]:
        """
        Score all indexed documents against query.
        Returns list of (doc_index, bm25_score).
        """
        query_tokens = self.tokenize(query)
        qf = Counter(query_tokens)

        scores = []
        for idx, doc_tokens in enumerate(self._doc_tokens):
            score = self._bm25_score(qf, doc_tokens)
            if score > 0:
                scores.append((idx, score))

        # Sort descending by score
        scores.sort(key=lambda x: -x[1])
        return scores

    def score_and_return(self, query: str, top_k: int = 50
                         ) -> List[Tuple[Dict[str, Any], float]]:
        """Score and return top-k papers with scores."""
        scored = self.score(query)
        return [(self._docs[idx], score) for idx, score in scored[:top_k]]

    def _bm25_score(self, qf: Counter, doc_tokens: List[str]) -> float:
        """Compute BM25 score for a single document."""
        dl = len(doc_tokens)
        doc_tf = Counter(doc_tokens)
        score = 0.0

        for term, tf_q in qf.items():
            df = self._df.get(term, 0)
            if df == 0:
                continue
            tf_d = doc_tf.get(term, 0)
            if tf_d == 0:
                continue

            # IDF
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))

            # TF saturation
            numerator = tf_d * (self.k1 + 1)
            denominator = tf_d + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
            tf_component = numerator / denominator

            score += idf * tf_component * tf_q  # Multiply by query TF

        return score

    def get_top_n(self, query: str, n: int = 50) -> List[Dict[str, Any]]:
        """Convenience: get top-N papers for a query."""
        return [doc for doc, _ in self.score_and_return(query, top_k=n)]


def build_bm25_from_papers(papers: List[Dict[str, Any]]) -> BM25Retriever:
    """Factory: create and fit a BM25Retriever from a list of paper dicts."""
    k1 = config.get("retrieval", "bm25", "k1", default=1.5)
    b = config.get("retrieval", "bm25", "b", default=0.75)
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.fit(papers)
    return retriever
