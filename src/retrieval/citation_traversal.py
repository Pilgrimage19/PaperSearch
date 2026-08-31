# ============================================================
# src/retrieval/citation_traversal.py - 引文扩展（batch 筛选高引节点 + 滚雪球）
# ============================================================

import logging
from typing import List, Dict, Any, Set
from src.utils.config_loader import config

logger = logging.getLogger("paper_search.citation")


def _safe_fetch(func, pid: str, limit: int) -> List[Dict[str, Any]]:
    """安全调用 S2 引文接口，失败返回空列表。"""
    try:
        return func(pid, limit=limit)
    except Exception as e:
        logger.debug(f"citation fetch failed for {pid}: {e}")
        return []


def _select_top_cited(
    papers: List[Dict[str, Any]], s2_client, top_cited: int
) -> List[Dict[str, Any]]:
    """
    用 batch 接口批量获取候选论文的引用数，只返回高引论文（引文遍历的关键节点）。
    一次 batch 请求查 N 篇，替代 N 次单篇查询，大幅减少调用量。
    """
    ids = [p.get("paper_id", "") for p in papers if p.get("paper_id")]
    if not ids:
        return []
    if len(ids) <= top_cited:
        return papers

    try:
        details = s2_client.get_papers_batch(ids)
        details.sort(key=lambda p: p.get("reference_count") or 0, reverse=True)
        top_ids = {p.get("paper_id", "") for p in details[:top_cited]}
        return [p for p in papers if p.get("paper_id", "") in top_ids]
    except Exception as e:
        logger.warning(f"batch cite-filter failed, fallback to first {top_cited}: {e}")
        return papers[:top_cited]


def _traverse_one_round(
    frontier: List[Dict[str, Any]],
    existing_ids: Set[str],
    s2_client,
    refs_per_paper: int,
    cites_per_paper: int,
) -> List[Dict[str, Any]]:
    """对 frontier 里的论文做一层后向（references）引文扩展，可选前向（citations）。"""
    new_papers: List[Dict[str, Any]] = []
    for p in frontier:
        pid = p.get("paper_id", "")
        if not pid:
            continue
        for ref in _safe_fetch(s2_client.get_references, pid, refs_per_paper):
            rid = ref.get("paper_id", "")
            if rid and rid not in existing_ids:
                existing_ids.add(rid)
                new_papers.append(ref)
        # cites_per_paper == 0 时跳过前向引文（调用减半）
        if cites_per_paper and cites_per_paper > 0:
            for cite in _safe_fetch(s2_client.get_citations, pid, cites_per_paper):
                cid = cite.get("paper_id", "")
                if cid and cid not in existing_ids:
                    existing_ids.add(cid)
                    new_papers.append(cite)
    return new_papers


def expand_with_citations(
    papers: List[Dict[str, Any]], s2_client
) -> List[Dict[str, Any]]:
    """
    引文扩展：先用 batch 接口筛出高引论文（关键节点），再对它们做前向+后向引文遍历，
    可选多轮滚雪球。相比逐篇遍历，调用量大幅下降且召回损失小。
    """
    enabled = config.get("retrieval", "citation_traversal", "enabled", default=True)
    if not enabled:
        return papers

    top_n = config.get("retrieval", "citation_traversal", "max_traversal_nodes", default=20)
    top_cited = config.get("retrieval", "citation_traversal", "top_cited_nodes", default=10)
    rounds = config.get("retrieval", "citation_traversal", "traversal_rounds", default=1)
    max_new = config.get("retrieval", "citation_traversal", "max_new_papers", default=200)
    refs_per_paper = config.get("retrieval", "citation_traversal", "refs_per_paper", default=50)
    cites_per_paper = config.get("retrieval", "citation_traversal", "cites_per_paper", default=50)

    all_papers = list(papers)
    existing_ids = {p.get("paper_id", "") for p in papers if p.get("paper_id")}

    frontier = _select_top_cited(papers[:top_n], s2_client, top_cited)
    for round_i in range(rounds):
        new_papers = _traverse_one_round(frontier, existing_ids, s2_client, refs_per_paper, cites_per_paper)
        if not new_papers:
            break
        all_papers.extend(new_papers)
        logger.info(f"  Citation round {round_i+1}: +{len(new_papers)} new papers")
        if len(all_papers) - len(papers) >= max_new:
            break
        frontier = _select_top_cited(new_papers[:top_n], s2_client, top_cited)

    added = len(all_papers) - len(papers)
    if added > max_new:
        all_papers = papers + all_papers[len(papers):len(papers) + max_new]
        added = max_new

    logger.info(f"  Citation traversal: {rounds} round(s), +{added} new papers")
    return all_papers
