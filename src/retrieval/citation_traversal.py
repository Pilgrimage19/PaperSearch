# ============================================================
# src/retrieval/citation_traversal.py - PageRank-guided citation expansion
# ============================================================

from typing import List, Dict, Any, Set, Optional
from collections import Counter
from src.utils.config_loader import config


def pagerank_citation_traversal(
    candidate_papers: List[Dict[str, Any]],
    s2_client,
    top_traversal_nodes: int = 5,
    max_new_papers: int = 20,
    alpha: float = 0.85,
) -> List[Dict[str, Any]]:
    """
    PageRank-guided citation traversal.

    1. Build local citation graph from top-N candidate papers
    2. Compute PageRank scores
    3. Traverse references of high-PageRank nodes
    4. Prioritize papers co-cited by multiple top papers

    Args:
        candidate_papers: Top papers after initial retrieval + ranking
        s2_client: Semantic Scholar client instance
        top_traversal_nodes: Number of high-PR nodes to traverse
        max_new_papers: Max new papers to add from expansion
        alpha: PageRank damping factor

    Returns:
        List of newly discovered papers
    """
    if len(candidate_papers) < 3:
        return []

    # Clear module-level storage from previous calls
    _ref_store.clear()

    # Use top-30 for graph construction to keep API calls manageable
    graph_papers = candidate_papers[:30]

    # 1. Build citation graph
    G, paper_ids = _build_local_citation_graph(graph_papers, s2_client)

    if len(G) < 3:
        return []

    # 2. Compute PageRank
    try:
        import networkx as nx
        pr = nx.pagerank(nx.DiGraph(G), alpha=alpha)
    except ImportError:
        # Fallback: simple degree-based centrality
        pr = _simple_centrality(G)

    # 3. Select top traversal nodes (only from our candidate pool)
    candidate_ids = {p.get("paper_id", "") for p in graph_papers}
    traversal_candidates = [
        (pid, score) for pid, score in pr.items()
        if pid in candidate_ids
    ]
    traversal_candidates.sort(key=lambda x: -x[1])
    top_nodes = [pid for pid, _ in traversal_candidates[:top_traversal_nodes]]

    # 4. Get references of top nodes
    ref_counter: Counter = Counter()
    for node_id in top_nodes:
        try:
            refs = s2_client.get_references(node_id, limit=50)
            for ref in refs:
                rid = ref.get("paper_id", "")
                if rid and rid not in candidate_ids:
                    ref_counter[rid] += 1
                    # Store paper data for later retrieval
                    _ref_store[rid] = ref
        except Exception:
            continue

    # 5. Prioritize: papers referenced by multiple top nodes first
    new_papers = []
    for ref_id, count in ref_counter.most_common(max_new_papers):
        paper = _ref_store.get(ref_id)
        if paper:
            new_papers.append(paper)

    return new_papers


# Module-level storage for reference papers
_ref_store: Dict[str, Dict[str, Any]] = {}


def _build_local_citation_graph(
    papers: List[Dict[str, Any]], s2_client
) -> tuple:
    """Build a local citation graph from paper list using S2 API."""
    edges = []
    paper_ids: Set[str] = set()

    for paper in papers[:15]:  # Limit API calls
        pid = paper.get("paper_id", "")
        if not pid:
            continue
        paper_ids.add(pid)

        # Get references (outgoing edges: pid -> ref_id)
        try:
            refs = s2_client.get_references(pid, limit=30)
            for ref in refs:
                rid = ref.get("paper_id", "")
                if rid:
                    paper_ids.add(rid)
                    edges.append((pid, rid))
        except Exception:
            pass

        # Get citations (incoming edges: citing_id -> pid)
        try:
            cites = s2_client.get_citations(pid, limit=30)
            for cite in cites:
                cid = cite.get("paper_id", "")
                if cid:
                    paper_ids.add(cid)
                    edges.append((cid, pid))
        except Exception:
            pass

    return edges, paper_ids


def _simple_centrality(edges) -> Dict[str, float]:
    """Fallback centrality: count of connections for each node."""
    centrality: Counter = Counter()
    for src, dst in edges:
        centrality[src] += 1
        centrality[dst] += 1
    total = max(sum(centrality.values()), 1)
    return {k: v / total for k, v in centrality.items()}


def expand_with_citations(
    papers: List[Dict[str, Any]], s2_client
) -> List[Dict[str, Any]]:
    """
    Convenience: expand candidate pool with citation traversal.
    Returns original papers + newly discovered papers.
    """
    enabled = config.get("retrieval", "citation_traversal", "enabled", default=True)
    if not enabled:
        return papers

    top_n = config.get("retrieval", "citation_traversal", "max_traversal_nodes", default=5)
    max_new = config.get("retrieval", "citation_traversal", "max_new_papers", default=20)
    alpha = config.get("retrieval", "citation_traversal", "pagerank_alpha", default=0.85)

    new_papers = pagerank_citation_traversal(
        papers, s2_client,
        top_traversal_nodes=top_n,
        max_new_papers=max_new,
        alpha=alpha,
    )

    # Deduplicate by paper_id
    existing_ids = {p.get("paper_id", "") for p in papers}
    truly_new = [p for p in new_papers if p.get("paper_id", "") not in existing_ids]

    return papers + truly_new
