# ============================================================
# src/retrieval/citation_traversal.py - 引文扩展（广度优先，前向+后向）
# ============================================================

import logging
from typing import List, Dict, Any
from src.utils.config_loader import config

logger = logging.getLogger("paper_search.citation")


def _safe_fetch(func, pid: str, limit: int) -> List[Dict[str, Any]]:
    """安全调用 S2 引文接口，失败返回空列表。"""
    try:
        return func(pid, limit=limit)
    except Exception as e:
        logger.debug(f"citation fetch failed for {pid}: {e}")
        return []


def expand_with_citations(
    papers: List[Dict[str, Any]], s2_client
) -> List[Dict[str, Any]]:
    """
    对 top-N 论文做一层前向（被引）+ 后向（引用）引文扩展。

    相关论文互相引用，引文遍历能召回关键词搜索找不到的论文。
    参考 PaSa 的 [Expand]、SPAR 的 RefChain、PaperQA2 的 Citation Traversal。

    Returns:
        原论文 + 新发现的引文论文
    """
    enabled = config.get("retrieval", "citation_traversal", "enabled", default=True)
    if not enabled:
        return papers

    top_n = config.get("retrieval", "citation_traversal", "max_traversal_nodes", default=20)
    max_new = config.get("retrieval", "citation_traversal", "max_new_papers", default=50)
    refs_per_paper = config.get("retrieval", "citation_traversal", "refs_per_paper", default=30)
    cites_per_paper = config.get("retrieval", "citation_traversal", "cites_per_paper", default=30)

    existing_ids = {p.get("paper_id", "") for p in papers if p.get("paper_id")}
    new_papers: List[Dict[str, Any]] = []
    seen_new = set()

    for p in papers[:top_n]:
        pid = p.get("paper_id", "")
        if not pid:
            continue

        # 后向引文：这篇论文引用了哪些论文（能召回基础/经典工作）
        for ref in _safe_fetch(s2_client.get_references, pid, refs_per_paper):
            rid = ref.get("paper_id", "")
            if rid and rid not in existing_ids and rid not in seen_new:
                seen_new.add(rid)
                new_papers.append(ref)

        # 前向引文：谁引用了这篇论文（能召回后续工作）
        for cite in _safe_fetch(s2_client.get_citations, pid, cites_per_paper):
            cid = cite.get("paper_id", "")
            if cid and cid not in existing_ids and cid not in seen_new:
                seen_new.add(cid)
                new_papers.append(cite)

    added = new_papers[:max_new]
    logger.info(f"  Citation traversal: traversed {min(len(papers), top_n)} papers, +{len(added)} new")
    return papers + added
