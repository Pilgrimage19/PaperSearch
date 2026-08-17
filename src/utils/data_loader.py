# ============================================================
# src/utils/data_loader.py - Dataset loading for PaSa & Asta
# ============================================================

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_pasa_dataset(path: str, max_queries: int = 0,
                      query_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load RealScholarQuery (PaSa) dataset.
    Returns list of dicts with keys: question, answer, answer_arxiv_id, qid, source_meta.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PaSa dataset not found: {path}")

    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))

    # Filter by specific IDs
    if query_ids:
        queries = [q for q in queries if q["qid"] in query_ids]

    # Limit
    if max_queries and max_queries > 0:
        queries = queries[:max_queries]

    return queries


def load_asta_dataset(test_path: str, val_path: Optional[str] = None,
                      query_types: Optional[List[str]] = None,
                      max_queries: int = 0) -> List[Dict[str, Any]]:
    """
    Load PaperFindingBench (Asta) dataset.
    query_types: ['semantic', 'specific', 'metadata'] or None for all.
    """
    all_queries = []
    for p in [test_path] + ([val_path] if val_path else []):
        p = Path(p)
        if not p.exists():
            print(f"[WARN] Asta file not found, skipping: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_queries.extend(data)

    # Filter by type
    if query_types:
        def qtype(q: dict) -> str:
            return q["input"]["query_id"].split("_")[0]
        all_queries = [q for q in all_queries if qtype(q) in query_types]

    # Limit
    if max_queries and max_queries > 0:
        all_queries = all_queries[:max_queries]

    return all_queries


def load_asta_normalizer(path: str) -> Dict[str, int]:
    """Load Asta normalizer reference (estimated set sizes for semantic queries)."""
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Normalizer not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_query_text(item: Dict[str, Any]) -> str:
    """Extract natural language query string from a dataset item."""
    if "question" in item:
        return item["question"].strip()          # PaSa format
    elif "input" in item:
        return item["input"]["query"].strip()    # Asta format
    return ""


def get_query_id(item: Dict[str, Any]) -> str:
    """Extract query ID from a dataset item."""
    if "qid" in item:
        return item["qid"]                       # PaSa: "RealScholarQuery_0"
    elif "input" in item:
        return item["input"]["query_id"]         # Asta: "semantic_1"
    return ""


def get_query_type(item: Dict[str, Any]) -> str:
    """Determine query type: semantic, specific, or metadata."""
    qid = get_query_id(item)
    if qid.startswith("semantic") or qid.startswith("RealScholarQuery"):
        return "semantic"
    elif qid.startswith("specific"):
        return "specific"
    elif qid.startswith("metadata"):
        return "metadata"
    return "semantic"


def get_gold_answers(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get ground truth answers in a unified format.
    Returns dict with keys depending on dataset:
      - pasa: {arxiv_ids: [...], titles: [...]}
      - asta semantic: {known_good_ids: [...], criteria: [...]}
      - asta specific/metadata: {corpus_ids: [...]}
    """
    if "answer_arxiv_id" in item:
        # PaSa format
        return {
            "arxiv_ids": item.get("answer_arxiv_id", []),
            "titles": item.get("answer", []),
            "type": "pasa",
        }
    else:
        # Asta format
        sc = item.get("scorer_criteria", {})
        qtype = get_query_type(item)
        return {
            "known_good_ids": sc.get("known_to_be_good", []),
            "known_bad_ids": sc.get("known_to_be_bad", []),
            "corpus_ids": sc.get("corpus_ids", []),
            "criteria": sc.get("relevance_criteria", []),
            "type": f"asta_{qtype}",
        }
