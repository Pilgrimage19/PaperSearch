# ============================================================
# src/step3_ranking.py
# Module 3: Paper Ranking (Coarse multi-dimension + Fine LLM rerank)
# ============================================================

import json
import math
import re
import logging
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config
from src.llm.client import LLMClient
from src.prompts import FINE_RANK_SYSTEM, FINE_RANK_PROMPT

logger = logging.getLogger("paper_search.step3")


# ---- Coarse Ranking (Zero LLM) ----

def coarse_rank(
    candidates: List[Dict[str, Any]],
    query: str,
    constraints: List[Dict[str, Any]],
    current_year: int = 2026,
) -> List[Dict[str, Any]]:
    """
    Multi-dimensional coarse scoring - no LLM.
    Computes 6 dimension scores and weighted sum.
    """
    weights = config.get("ranking", "coarse", "weights", default={
        "bm25_relevance": 0.25,
        "embedding_similarity": 0.25,
        "citation_count": 0.15,
        "recency": 0.15,
        "venue_prestige": 0.10,
        "constraint_coverage": 0.10,
    })

    top_venues = set(config.get("ranking", "coarse", "top_venues", default=[
        "ACL", "EMNLP", "NAACL", "NeurIPS", "ICML", "ICLR",
        "CVPR", "ICCV", "AAAI", "IJCAI", "SIGIR", "WWW", "KDD",
        "Nature", "Science",
    ]))
    decay_factor = config.get("ranking", "coarse", "recency_decay_factor", default=3.0)

    # Compute max citation count for normalization
    max_citations = max((p.get("citation_count", 0) for p in candidates), default=1)

    scored = []
    for paper in candidates:
        scores = {}

        # 1. BM25 Relevance: compute locally
        scores["bm25_relevance"] = _simple_bm25_score(query, paper)

        # 2. Embedding Similarity: compute locally
        scores["embedding_similarity"] = _simple_embedding_score(query, paper)

        # 3. Citation Count (log-normalized)
        cites = paper.get("citation_count", 0) or 0
        scores["citation_count"] = math.log(1 + cites) / math.log(1 + max(max_citations, 1))

        # 4. Recency (exponential decay)
        year = paper.get("year") or current_year
        age = max(current_year - year, 0)
        scores["recency"] = math.exp(-age / decay_factor)

        # 5. Venue Prestige (binary: top venue or not)
        venue = paper.get("venue") or ""
        scores["venue_prestige"] = 1.0 if any(v in venue for v in top_venues) else 0.3

        # 6. Constraint Coverage
        scores["constraint_coverage"] = _constraint_coverage(query, paper, constraints)

        # Weighted sum
        total_score = sum(weights.get(k, 0.1) * scores.get(k, 0.0) for k in weights)

        paper["_coarse_scores"] = scores
        paper["_coarse_total"] = round(total_score, 4)

        scored.append(paper)

    # Sort by coarse score descending
    scored.sort(key=lambda p: p.get("_coarse_total", 0), reverse=True)
    return scored


def _simple_bm25_score(query: str, paper: Dict[str, Any]) -> float:
    """Compute a simple BM25-like score between query and paper title+abstract."""
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    query_terms = set(query.lower().split())
    if not query_terms:
        return 0.0
    # Simple term match ratio with IDF-like weighting
    matches = sum(1 for t in query_terms if t in text)
    return min(matches / len(query_terms), 1.0)


def _simple_embedding_score(query: str, paper: Dict[str, Any]) -> float:
    """Compute a simple embedding-like score using word overlap."""
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    text_words = set(re.findall(r"[a-z0-9]+", text))
    if not query_words:
        return 0.0
    overlap = query_words & text_words
    return len(overlap) / len(query_words)


def _constraint_coverage(
    query: str, paper: Dict[str, Any], constraints: List[Dict[str, Any]]
) -> float:
    """Check what fraction of query constraints are covered."""
    if not constraints:
        return 0.5
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    covered = 0
    for c in constraints:
        value = (c.get("value") or "").lower()
        if not value:
            covered += 0.5
            continue
        # Simple keyword matching
        keywords = set(re.findall(r"[a-z0-9]+", value))
        if keywords:
            match_ratio = sum(1 for k in keywords if k in text) / len(keywords)
            covered += match_ratio
    return covered / len(constraints)


# ---- Fine Ranking (LLM API) ----


def _finalize_paper(paper: Dict[str, Any], llm_score: Optional[float] = None) -> None:
    """
    Assign _final_score and _relevance_layer to a single paper.
    If llm_score is None, use pure coarse score (mapped to 0-100).
    """
    coarse_w = config.get("ranking", "fine", "final_score", "coarse_weight", default=0.4)
    llm_w = config.get("ranking", "fine", "final_score", "llm_weight", default=0.6)
    layers = config.get("organization", "relevance_layers", default={
        "highly_relevant": 70,
        "partially_relevant": 40,
        "background": 20,
    })

    coarse = paper.get("_coarse_total") or 0
    if llm_score is None:
        # 未经过 LLM 精排的论文，final_score 直接用粗排分数映射到 0-100
        paper["_final_score"] = round(coarse * 100, 2)
    else:
        paper["_final_score"] = round(coarse_w * coarse * 100 + llm_w * llm_score, 2)

    fs = paper["_final_score"]
    if fs >= layers.get("highly_relevant", 70):
        paper["_relevance_layer"] = "highly_relevant"
    elif fs >= layers.get("partially_relevant", 40):
        paper["_relevance_layer"] = "partially_relevant"
    elif fs >= layers.get("background", 20):
        paper["_relevance_layer"] = "background"
    else:
        paper["_relevance_layer"] = "irrelevant"


def fine_rank(
    papers: List[Dict[str, Any]],
    query: str,
    constraints: List[Dict[str, Any]],
    client: Optional[LLMClient] = None,
) -> List[Dict[str, Any]]:
    """
    LLM-based fine-grained relevance ranking.
    Papers are batch-processed to minimize API calls.
    """
    if not papers:
        return papers

    enabled = config.get("ranking", "fine", "enabled", default=True)
    if not enabled:
        return papers

    if client is None:
        client = _make_fine_client()

    batch_size = config.get("ranking", "fine", "batch_size", default=5)
    constraints_json = json.dumps(constraints, ensure_ascii=False) if constraints else "[]"

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]

        # Format paper list
        paper_lines = []
        for j, paper in enumerate(batch):
            paper_lines.append(
                f"Paper {j}: "
                f"Title: {paper.get('title', 'N/A')}\n"
                f"  Abstract: {(paper.get('abstract') or 'N/A')[:500]}\n"
                f"  Year: {paper.get('year', 'N/A')}, "
                f"Venue: {paper.get('venue', 'N/A')}, "
                f"Citations: {paper.get('citation_count', 0)}"
            )

        prompt = FINE_RANK_PROMPT.format(
            query=query,
            constraints_json=constraints_json,
            paper_list="\n\n".join(paper_lines),
        )

        try:
            batch_scores = client.chat_json(prompt, system=FINE_RANK_SYSTEM, repair_json=True)
            if isinstance(batch_scores, dict):
                batch_scores = [batch_scores]
            for item in batch_scores:
                idx = item.get("paper_index", 0)
                if idx < len(batch):
                    batch[idx]["_llm_score"] = item.get("overall_score", 50)
                    batch[idx]["_constraint_scores"] = item.get("constraint_scores", {})
                    batch[idx]["_evidence"] = item.get("evidence", "")
                    batch[idx]["_relevance_level"] = item.get("relevance_level", "partially_relevant")
        except Exception as e:
            logger.warning(f"Fine ranking batch failed: {e}")
            for paper in batch:
                paper["_llm_score"] = paper.get("_coarse_total", 0) * 50
                paper["_evidence"] = ""

    # ---- Combine Coarse + Fine ----
    for paper in papers:
        _finalize_paper(paper, llm_score=paper.get("_llm_score"))

    papers.sort(key=lambda p: p.get("_final_score", 0), reverse=True)
    return papers


# ---- Main Step 3 Entry Point ----

def step3_rank_all(
    retrieval_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Run ranking for all retrieved query results.

    Args:
        retrieval_results: Output from step2_retrieve_all()

    Returns:
        Same structure with scored + ranked candidates.
    """
    top_k_after_coarse = config.get("ranking", "coarse", "top_k_after_coarse", default=15)

    results = []
    total = len(retrieval_results)

    for i, rr in enumerate(retrieval_results):
        qid = rr["qid"]
        query = rr["query"]
        candidates = rr.get("candidates", [])
        parsed = rr.get("parsed", {})
        constraints = parsed.get("constraints", [])

        logger.info(f"({i+1}/{total}) Ranking: {qid} ({len(candidates)} candidates)")

        # Coarse rank
        coarse_ranked = coarse_rank(candidates, query, constraints)
        if coarse_ranked:
            logger.info(f"  Coarse top score: {coarse_ranked[0].get('_coarse_total', 0):.3f}")
        else:
            logger.info(f"  No candidates")

        # Take top-N for fine ranking
        to_fine_rank = coarse_ranked[:top_k_after_coarse]

        # Fine rank
        fine_ranked = fine_rank(to_fine_rank, query, constraints)

        # Combine: fine-ranked top-N + remaining coarse-ranked
        fine_ids = {p.get("paper_id", ""): True for p in fine_ranked}
        remaining = [p for p in coarse_ranked if p.get("paper_id", "") not in fine_ids]

        # remaining 论文未经过 LLM 精排，补上纯粗排分数和分层
        for p in remaining:
            _finalize_paper(p, llm_score=None)

        final_ranked = fine_ranked + remaining
        final_ranked.sort(key=lambda p: p.get("_final_score", 0), reverse=True)

        results.append({
            "qid": rr["qid"],
            "query": rr["query"],
            "query_type": rr["query_type"],
            "parsed": rr.get("parsed", {}),
            "metadata": rr.get("metadata", {}),
            "ranked_papers": final_ranked,
        })

    return results


def _make_fine_client() -> LLMClient:
    """Create an LLMClient for fine ranking from config."""
    return LLMClient(
        model=config.get("ranking", "fine", "model", default="deepseek-v4-flash"),
        provider=config.get("ranking", "fine", "model_provider", default="deepseek"),
        temperature=config.get("ranking", "fine", "temperature", default=0.0),
        max_tokens=config.get("ranking", "fine", "max_tokens", default=3000),
    )
