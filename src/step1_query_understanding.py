# ============================================================
# src/step1_query_understanding.py
# Module 1: Query Understanding & Decomposition (1× LLM API call)
# ============================================================

import logging
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config
from src.utils.data_loader import get_query_text, get_query_id, get_query_type
from src.llm.client import LLMClient
from src.prompts import (
    QUERY_UNDERSTANDING_SYSTEM,
    QUERY_UNDERSTANDING_PROMPT,
)

logger = logging.getLogger("paper_search.step1")


def parse_query(
    query: str,
    client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """
    Parse a single academic query into structured search plan.
    Makes 1 LLM API call per query.
    """
    if client is None:
        client = _make_client()

    prompt = QUERY_UNDERSTANDING_PROMPT.format(user_query=query)

    try:
        parsed = client.chat_json(
            prompt,
            system=QUERY_UNDERSTANDING_SYSTEM,
            repair_json=True,
        )
    except Exception as e:
        logger.warning(f"Query understanding failed for '{query[:80]}...': {e}")
        parsed = _fallback_parse(query)

    # Ensure required fields exist
    parsed.setdefault("query_type", "semantic")
    parsed.setdefault("intent", "find")
    parsed.setdefault("domain", "AI")
    parsed.setdefault("constraints", [])
    parsed.setdefault("excluded", [])
    parsed.setdefault("sub_queries", {"keyword": [], "nl": []})
    parsed.setdefault("time_range", None)
    parsed.setdefault("preferred_sources", ["semantic_scholar", "arxiv"])

    # Ensure at least one keyword query
    if not parsed["sub_queries"].get("keyword"):
        parsed["sub_queries"]["keyword"] = [query]
    if not parsed["sub_queries"].get("nl"):
        parsed["sub_queries"]["nl"] = [query]

    return parsed


def _fallback_parse(query: str) -> Dict[str, Any]:
    """Fallback: simple rules when LLM fails."""
    return {
        "query_type": "semantic",
        "intent": "find",
        "domain": "AI",
        "constraints": [],
        "excluded": [],
        "sub_queries": {"keyword": [query], "nl": [query]},
        "time_range": None,
        "preferred_sources": ["semantic_scholar", "arxiv"],
    }


def step1_process_all(
    dataset_items: List[Dict[str, Any]],
    client: Optional[LLMClient] = None,
) -> List[Dict[str, Any]]:
    """
    Process all queries through query understanding.

    Returns:
        List of {qid, query, query_type, parsed: {...}, original_item: {...}}
    """
    if client is None:
        client = _make_client()

    results = []
    total = len(dataset_items)

    for i, item in enumerate(dataset_items):
        query_text = get_query_text(item)
        qid = get_query_id(item)
        qtype = get_query_type(item)

        logger.info(f"({i+1}/{total}) Understanding: {qid}")

        # For specific and metadata queries, skip LLM — use rules
        if qtype in ("specific", "metadata"):
            parsed = {
                "query_type": qtype,
                "intent": "find",
                "domain": "CS",
                "constraints": [],
                "excluded": [],
                "sub_queries": {"keyword": [query_text], "nl": []},
                "time_range": None,
                "preferred_sources": ["semantic_scholar"],
            }
        else:
            parsed = parse_query(query_text, client)

        results.append({
            "qid": qid,
            "query": query_text,
            "query_type": qtype,
            "parsed": parsed,
            "original_item": item,
        })

    return results


def _make_client() -> LLMClient:
    """Create an LLMClient from config settings."""
    return LLMClient(
        model=config.get("query_understanding", "model", default="deepseek-v4-flash"),
        provider=config.get("query_understanding", "model_provider", default="dashscope"),
        temperature=config.get("query_understanding", "temperature", default=0.0),
        max_tokens=config.get("query_understanding", "max_tokens", default=2000),
    )
