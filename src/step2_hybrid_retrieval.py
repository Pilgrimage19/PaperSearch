# ============================================================
# src/step2_hybrid_retrieval.py
# Module 2: 混合检索（BM25 + Embedding + 引文遍历）+ 迭代检索
# ============================================================

import logging
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config
from src.sources.semantic_scholar import s2_client
from src.sources.arxiv import arxiv_client
from src.sources.openalex import openalex_client
from src.retrieval.bm25 import build_bm25_from_papers
from src.retrieval.embedding import build_embedding_index
from src.retrieval.fusion import fuse_results
from src.retrieval.citation_traversal import expand_with_citations
from src.step2_critic import critique

logger = logging.getLogger("paper_search.step2")


def _retrieve_round(
    kw_queries: List[str],
    nl_queries: List[str],
    time_range: Optional[Dict[str, str]],
    source_list: List[str],
    per_source_k: int,
    pool_size: int,
    fusion_method: str,
):
    """
    单轮检索：Phase 1 召回 → Phase 2 BM25+Embedding 融合 → Phase 3 引文扩展。
    返回 (expanded_candidates, all_papers, filtered_papers)。
    """
    # ---- Phase 1: Broad Recall ----
    all_papers: List[Dict[str, Any]] = []
    seen_ids = set()
    for source_name in source_list:
        source_papers = _retrieve_from_source(
            source_name, kw_queries, nl_queries, per_source_k, time_range
        )
        for p in source_papers:
            pid = p.get("paper_id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)

    # ---- Phase 2: BM25 + Embedding 融合 ----
    if all_papers:
        bm25 = build_bm25_from_papers(all_papers)
        emb = build_embedding_index(all_papers)
        bm25_ranked = []
        emb_ranked = []
        for sq in kw_queries[:3]:
            bm25_ranked.append(bm25.score_and_return(sq, top_k=pool_size))
        for sq in nl_queries[:2]:
            emb_ranked.append(emb.search(sq, top_k=pool_size))
        ranked_lists = bm25_ranked + emb_ranked
        filtered = fuse_results(ranked_lists, method=fusion_method, top_k=pool_size)
    else:
        filtered = []

    # ---- Phase 3: Citation Expansion ----
    expanded = expand_with_citations(filtered, s2_client)
    return expanded, all_papers, filtered


def _merge_dedup(
    papers_a: List[Dict[str, Any]], papers_b: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """按 paper_id 去重合并两个论文列表。"""
    merged = list(papers_a)
    seen = {p.get("paper_id", "") for p in merged if p.get("paper_id")}
    for p in papers_b:
        pid = p.get("paper_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            merged.append(p)
    return merged


def step2_retrieve_all(
    parsed_queries: List[Dict[str, Any]],
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    迭代检索：先初始检索，再 Critic 自省检查约束覆盖，针对缺口补检索，循环 N 轮。
    """
    source_list = sources or config.get("retrieval", "sources",
                                         default=["semantic_scholar"])
    per_source_k = config.get("retrieval", "per_source_top_k", default=100)
    pool_size = config.get("retrieval", "candidate_pool_size", default=200)
    fusion_method = config.get("retrieval", "fusion", "method", default="rrf")
    iterative_enabled = config.get("retrieval", "iterative", "enabled", default=True)
    max_rounds = config.get("retrieval", "iterative", "max_rounds", default=3)

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
        constraints = parsed.get("constraints", [])
        time_range = parsed.get("time_range")

        logger.info(f"({i+1}/{total}) Retrieving: {qid} (type={qtype})")

        # ---- 第 1 轮：初始检索 ----
        candidates, all_papers, filtered = _retrieve_round(
            kw_queries, nl_queries, time_range, source_list,
            per_source_k, pool_size, fusion_method,
        )
        logger.info(f"  Round 1: {len(candidates)} candidates")

        # ---- 迭代检索：Critic 自省 + 补检索 ----
        used_queries = set(kw_queries + nl_queries)
        if iterative_enabled and constraints:
            for round_i in range(1, max_rounds):
                crit = critique(query, constraints, candidates)
                verdict = crit.get("verdict", "complete")
                if verdict == "complete":
                    logger.info(f"  Round {round_i+1} (critic): coverage complete, stop")
                    break

                new_queries = [
                    q for q in crit.get("new_queries", [])
                    if q and q.strip() and q.strip() not in used_queries
                ]
                if not new_queries:
                    logger.info(f"  Round {round_i+1} (critic): no new queries, stop")
                    break

                used_queries.update(q.strip() for q in new_queries)
                logger.info(f"  Round {round_i+1} (critic): +{len(new_queries)} gap queries -> {new_queries[:2]}")

                new_candidates, new_all, new_filtered = _retrieve_round(
                    new_queries, [], time_range, source_list,
                    per_source_k, pool_size, fusion_method,
                )
                candidates = _merge_dedup(candidates, new_candidates)
                all_papers = _merge_dedup(all_papers, new_all)
                filtered = _merge_dedup(filtered, new_filtered)
                logger.info(f"  Round {round_i+1}: merged {len(candidates)} candidates")

        results.append({
            "qid": qid,
            "query": query,
            "query_type": qtype,
            "candidates": candidates,
            "phase1_papers": [
                {"paper_id": p.get("paper_id", ""), "arxiv_id": p.get("arxiv_id", ""),
                 "title": p.get("title", "")}
                for p in all_papers
            ],
            "phase2_papers": [
                {"paper_id": p.get("paper_id", ""), "arxiv_id": p.get("arxiv_id", ""),
                 "title": p.get("title", "")}
                for p in filtered
            ],
            "metadata": {
                "api_recall_count": len(all_papers),
                "after_fusion_count": len(filtered),
                "after_expansion_count": len(candidates),
                "sources_used": source_list,
            },
        })

    return results


def _enrich_arxiv_ids(
    papers: List[Dict[str, Any]],
    max_enrich: int = 20,
) -> int:
    """
    对 arxiv_id 为空的论文，用标题去 arXiv 查询补全 arxiv_id。
    遇到 arXiv 限流（429）会抛异常，立即终止避免刷屏。返回成功补全的论文数。
    """
    enriched = 0
    for p in papers:
        if enriched >= max_enrich:
            break
        if p.get("arxiv_id"):
            continue
        title = p.get("title") or ""
        if not title:
            continue
        try:
            arxiv_id = arxiv_client.search_by_title(title)
        except Exception as e:
            logger.warning(f"arXiv enrich aborted (likely rate limit): {e}")
            break
        if arxiv_id:
            p["arxiv_id"] = arxiv_id
            enriched += 1
    return enriched


def _retrieve_from_source(
    source_name: str,
    kw_queries: List[str],
    nl_queries: List[str],
    per_source_k: int,
    time_range: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """从单个源检索论文。Semantic Scholar 用 kw+nl，其他源只用 kw。"""
    year_start = int(time_range["start"]) if time_range and time_range.get("start") else None
    year_end = int(time_range["end"]) if time_range and time_range.get("end") else None

    if source_name == "semantic_scholar":
        queries = (kw_queries[:5] + nl_queries[:2])
    else:
        queries = kw_queries[:5]

    all_papers = []
    for q in queries:
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
                papers = []
            elif source_name == "dblp":
                papers = []
            else:
                papers = []

            all_papers.extend(papers)
        except Exception as e:
            logger.warning(f"{source_name} search failed for '{q[:60]}...': {e}")

    return all_papers
