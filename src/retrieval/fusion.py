# ============================================================
# src/retrieval/fusion.py - Result fusion (RRF / CombSUM)
# ============================================================

from typing import List, Dict, Any, Tuple
from src.utils.config_loader import config


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Dict[str, Any], float]]],
    k: int = 60,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Reciprocal Rank Fusion (RRF).
    Each ranked_list is [(paper_dict, score), ...].
    Returns merged list sorted by RRF score descending.
    """
    scores: Dict[str, float] = {}
    paper_map: Dict[str, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, (paper, _) in enumerate(ranked):
            pid = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("title", "")
            if not pid:
                continue
            paper_map[pid] = paper
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    # Sort by RRF score
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return [(paper_map[pid], score) for pid, score in sorted_items]


def comb_sum_fusion(
    ranked_lists: List[List[Tuple[Dict[str, Any], float]]],
) -> List[Tuple[Dict[str, Any], float]]:
    """
    CombSUM fusion: sum normalized scores across lists.
    Requires min-max normalization per list first.
    """
    scores: Dict[str, float] = {}
    paper_map: Dict[str, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        if not ranked:
            continue
        # Min-max normalize scores for this list
        raw_scores = [s for _, s in ranked]
        s_min, s_max = min(raw_scores), max(raw_scores)
        score_range = s_max - s_min if s_max > s_min else 1.0

        for paper, raw_score in ranked:
            pid = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("title", "")
            if not pid:
                continue
            paper_map[pid] = paper
            normalized = (raw_score - s_min) / score_range
            scores[pid] = scores.get(pid, 0.0) + normalized

    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return [(paper_map[pid], score) for pid, score in sorted_items]


def fuse_results(
    ranked_lists: List[List[Tuple[Dict[str, Any], float]]],
    method: str = "rrf",
    top_k: int = 200,
) -> List[Dict[str, Any]]:
    """
    Unified fusion interface.
    Returns top_k papers (without scores).
    """
    if len(ranked_lists) == 0:
        return []
    if len(ranked_lists) == 1:
        return [p for p, _ in ranked_lists[0]][:top_k]

    if method == "rrf":
        k = config.get("retrieval", "fusion", "rrf_k", default=60)
        merged = reciprocal_rank_fusion(ranked_lists, k=k)
    elif method == "comb_sum":
        merged = comb_sum_fusion(ranked_lists)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    return [paper for paper, _ in merged[:top_k]]
