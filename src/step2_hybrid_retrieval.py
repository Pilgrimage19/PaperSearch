# ============================================================
# src/step2_hybrid_retrieval.py
# Module 2: Hybrid Retrieval (BM25 + Embedding + Citation Traversal)
# Zero LLM calls - pure retrieval
# ============================================================

import logging
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config

logger = logging.getLogger("paper_search.step2")
from src.sources.semantic_scholar import s2_client
from src.sources.arxiv import arxiv_client
from src.sources.openalex import openalex_client
from src.retrieval.bm25 import build_bm25_from_papers
from src.retrieval.embedding import build_embedding_index
from src.retrieval.fusion import fuse_results
from src.retrieval.citation_traversal import expand_with_citations


def step2_retrieve_all(
    parsed_queries: List[Dict[str, Any]],
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run hybrid retrieval for all parsed queries.

    Args:
        parsed_queries: Output from step1_process_all()
        sources: Override source list from config

    Returns:
        List of dicts: {qid, query, candidates: [...], metadata: {...}}
    """
    source_list = sources or config.get("retrieval", "sources",
                                         default=["semantic_scholar", "arxiv", "openalex"])
    per_source_k = config.get("retrieval", "per_source_top_k", default=50)
    pool_size = config.get("retrieval", "candidate_pool_size", default=200)
    fusion_method = config.get("retrieval", "fusion", "method", default="rrf")

    results = []
    total = len(parsed_queries)

    for i, pq in enumerate(parsed_queries):
        qid = pq["qid"]
        query = pq["query"]
        qtype = pq["query_type"]
        parsed = pq.get("parsed", {})
        sub_queries = parsed.get("sub_queries", {})
        kw_queries = sub_queries.get("keyword", [query])
        nl_queries = sub_queries.get("nl", [query])
        time_range = parsed.get("time_range")

        logger.info(f"({i+1}/{total}) Retrieving: {qid} (type={qtype})")

        # ---- Phase 1: Broad Recall ----
        all_papers: List[Dict[str, Any]] = []
        seen_ids = set()

        for source_name in source_list:
            source_papers = _retrieve_from_source(
                source_name, kw_queries, per_source_k, time_range
            )
            for p in source_papers:
                pid = p.get("paper_id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)

        logger.info(f"  Phase 1 (API recall): {len(all_papers)} papers from {len(source_list)} sources")

        # ---- Phase 2: Local BM25 + Embedding Re-ranking ----
        if len(all_papers) > 0:
            # Build local indexes
            bm25 = build_bm25_from_papers(all_papers)
            emb = build_embedding_index(all_papers)

            # Rank with each sub-query
            bm25_ranked = []
            emb_ranked = []
            for sq in kw_queries[:3]:  # Limit sub-queries for efficiency
                bm25_ranked.append(bm25.score_and_return(sq, top_k=pool_size))
            for sq in nl_queries[:2]:
                emb_ranked.append(emb.search(sq, top_k=pool_size))

            # Fuse results
            ranked_lists = bm25_ranked + emb_ranked
            filtered = fuse_results(ranked_lists, method=fusion_method, top_k=pool_size)
        else:
            filtered = []

        logger.info(f"  Phase 2 (BM25+Embedding fusion): {len(filtered)} candidates")

        # ---- Phase 3: Citation Expansion ----
        expanded = expand_with_citations(filtered, s2_client)
        if len(expanded) > len(filtered):
            logger.info(f"  Phase 3 (Citation traversal): +{len(expanded) - len(filtered)} papers")

        results.append({
            "qid": qid,
            "query": query,
            "query_type": qtype,
            "candidates": expanded,
            "metadata": {
                "api_recall_count": len(all_papers),
                "after_fusion_count": len(filtered),
                "after_expansion_count": len(expanded),
                "sources_used": source_list,
            },
        })

    return results


def _retrieve_from_source(
    source_name: str,
    kw_queries: List[str],
    per_source_k: int,
    time_range: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Retrieve papers from a single source for all keyword queries."""
    year_start = int(time_range["start"]) if time_range and time_range.get("start") else None
    year_end = int(time_range["end"]) if time_range and time_range.get("end") else None

    all_papers = []
    for q in kw_queries[:5]:  # Limit keyword queries per source
        try:
            if source_name == "semantic_scholar":
                papers = s2_client.keyword_search(
                    q, limit=per_source_k,
                    year_start=year_start, year_end=year_end,
                )
            elif source_name == "arxiv":
                papers = arxiv_client.search(
                    q, max_results=per_source_k,
                    year_start=year_start, year_end=year_end,
                )
            elif source_name == "openalex":
                papers = openalex_client.search_works(
                    q, limit=per_source_k,
                    year_start=year_start, year_end=year_end,
                )
            elif source_name == "pubmed":
                # PubMed client not implemented yet - skip
                papers = []
            elif source_name == "dblp":
                # DBLP client not implemented yet - skip
                papers = []
            else:
                papers = []

            all_papers.extend(papers)
        except Exception as e:
            logger.warning(f"{source_name} search failed for '{q[:60]}...': {e}")

    return all_papers
