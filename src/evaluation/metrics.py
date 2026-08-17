# ============================================================
# src/evaluation/metrics.py - Evaluation metrics computation
# ============================================================

import math
from typing import List, Dict, Any, Set


def precision(predicted_ids: Set[str], gold_ids: Set[str]) -> float:
    """Precision = |TP| / |predicted|."""
    if len(predicted_ids) == 0:
        return 0.0
    tp = len(predicted_ids & gold_ids)
    return tp / len(predicted_ids)


def recall(predicted_ids: Set[str], gold_ids: Set[str]) -> float:
    """Recall = |TP| / |gold|."""
    if len(gold_ids) == 0:
        return 1.0 if len(predicted_ids) == 0 else 0.0
    tp = len(predicted_ids & gold_ids)
    return tp / len(gold_ids)


def f1_score(predicted_ids: Set[str], gold_ids: Set[str]) -> float:
    """F1 = harmonic mean of precision and recall."""
    p = precision(predicted_ids, gold_ids)
    r = recall(predicted_ids, gold_ids)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def recall_at_k(predicted_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """Recall@K: proportion of gold items found in top-K predictions."""
    if len(gold_ids) == 0:
        return 1.0
    top_k = set(predicted_ids[:k])
    tp = len(top_k & gold_ids)
    return tp / len(gold_ids)


def ndcg_at_k(
    predicted_ids: List[str],
    gold_ids: Set[str],
    relevance_scores: Dict[str, float],
    k: int = 10,
) -> float:
    """nDCG@K with graded relevance."""
    if not predicted_ids or not gold_ids:
        return 0.0

    # DCG
    dcg = 0.0
    for i, pid in enumerate(predicted_ids[:k]):
        rel = relevance_scores.get(pid, 0.0)
        dcg += rel / math.log2(i + 2)  # i is 0-indexed

    # IDCG (ideal: all gold items at top with max relevance)
    ideal_rels = sorted(
        [relevance_scores.get(gid, 1.0) for gid in gold_ids],
        reverse=True,
    )[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_pasa(
    ranked_papers: List[Dict[str, Any]],
    gold_data: Dict[str, Any],
    ks: List[int] = [5, 10, 20, 50],
) -> Dict[str, float]:
    """
    Evaluate against PaSa (RealScholarQuery) ground truth.
    Gold data has arxiv_ids as answer keys.
    """
    gold_ids = set(gold_data.get("arxiv_ids", []))
    if not gold_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Extract predicted arXiv IDs in ranked order
    pred_ids = []
    for p in ranked_papers:
        aid = p.get("arxiv_id", "")
        if not aid:
            # Try to extract from external IDs or paper_id
            aid = p.get("paper_id", "")
        if aid:
            pred_ids.append(aid)

    pred_set = set(pred_ids)

    metrics = {
        "precision": round(precision(pred_set, gold_ids), 4),
        "recall": round(recall(pred_set, gold_ids), 4),
        "f1": round(f1_score(pred_set, gold_ids), 4),
    }

    for k in ks:
        metrics[f"recall_at_{k}"] = round(recall_at_k(pred_ids, gold_ids, k), 4)

    return metrics


def evaluate_asta_semantic(
    ranked_papers: List[Dict[str, Any]],
    gold_data: Dict[str, Any],
    normalizer_value: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate against Asta semantic query.
    Uses known_good as seed and normalizer for estimated recall.
    """
    known_good = set(gold_data.get("known_good_ids", []))
    criteria = gold_data.get("criteria", [])

    if not criteria:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_ids = [p.get("paper_id", "") for p in ranked_papers if p.get("paper_id")]

    # Precision against known good
    pred_set = set(pred_ids)
    tp_known = len(pred_set & known_good)

    prec = tp_known / max(len(pred_set), 1)

    # Estimated recall using normalizer
    if normalizer_value and normalizer_value > 0:
        est_recall = tp_known / normalizer_value
    else:
        est_recall = tp_known / max(len(known_good), 1)

    f1 = 2 * prec * est_recall / (prec + est_recall) if (prec + est_recall) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "estimated_recall": round(est_recall, 4),
        "f1": round(f1, 4),
        "known_good_found": tp_known,
        "known_good_total": len(known_good),
        "normalizer_value": normalizer_value or 0,
    }


def evaluate_asta_exact(
    ranked_papers: List[Dict[str, Any]],
    gold_data: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate against Asta specific/metadata queries (exact match)."""
    corpus_ids = set(gold_data.get("corpus_ids", []))
    if not corpus_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_ids = [p.get("paper_id", "") for p in ranked_papers if p.get("paper_id")]
    pred_set = set(pred_ids)

    return {
        "precision": round(precision(pred_set, corpus_ids), 4),
        "recall": round(recall(pred_set, corpus_ids), 4),
        "f1": round(f1_score(pred_set, corpus_ids), 4),
    }
