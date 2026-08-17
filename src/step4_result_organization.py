# ============================================================
# src/step4_result_organization.py
# Module 4: Result Organization & Structured Output
# ============================================================

import json
import os
import logging
from typing import List, Dict, Any, Optional
from src.utils.config_loader import config
from src.llm.client import LLMClient
from src.prompts import COMPARISON_TABLE_SYSTEM, COMPARISON_TABLE_PROMPT

logger = logging.getLogger("paper_search.step4")


def step4_organize_all(
    ranking_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Organize ranked results into structured output.

    Per query produces:
    - Layered paper list (highly/partially/background)
    - Constraint satisfaction matrix
    - Method comparison table (LLM, top-5)
    - Citation graph data
    """

    total = len(ranking_results)
    for i, rr in enumerate(ranking_results):
        qid = rr["qid"]
        query = rr["query"]
        papers = rr.get("ranked_papers", [])

        logger.info(f"({i+1}/{total}) Organizing: {qid} ({len(papers)} papers)")

        # 1. Layer papers
        layers = _layer_papers(papers)

        # 2. Constraint matrix
        matrix = _build_constraint_matrix(papers, rr.get("parsed", {}))

        # 3. Comparison table (LLM call for top-5)
        top5 = [p for p in papers[:5] if p.get("_final_score", 0) > 40]
        comp_table = _build_comparison_table(top5, query) if top5 else []

        # 4. Citation graph data
        citation_graph = _build_citation_graph_data(papers[:15])

        rr["structured_output"] = {
            "layers": layers,
            "constraint_matrix": matrix,
            "comparison_table": comp_table,
            "citation_graph": citation_graph,
        }

    return ranking_results


def _layer_papers(papers: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """Group papers by relevance layer."""
    layers = {
        "highly_relevant": [],
        "partially_relevant": [],
        "background": [],
    }
    for p in papers:
        layer = p.get("_relevance_layer", "partially_relevant")
        if layer in layers:
            layers[layer].append({
                "title": p.get("title") or "",
                "year": p.get("year"),
                "venue": p.get("venue"),
                "score": p.get("_final_score") or 0,
                "evidence": p.get("_evidence") or "",
                "arxiv_id": p.get("arxiv_id") or "",
                "citation_count": p.get("citation_count") or 0,
                "url": p.get("url") or "",
            })
    return layers


def _build_constraint_matrix(
    papers: List[Dict[str, Any]], parsed: Dict[str, Any]
) -> Dict[str, Any]:
    """Build constraint × paper satisfaction matrix."""
    constraints = parsed.get("constraints", [])
    if not constraints:
        return {"constraints": [], "matrix": []}

    matrix = []
    for p in papers[:20]:
        cs = p.get("_constraint_scores", {})
        row = {
            "title": (p.get("title") or "")[:80],
            "score": p.get("_final_score"),
        }
        for c in constraints:
            name = c.get("name", c.get("value", "?"))
            row[name] = cs.get(name, "-")
        matrix.append(row)

    return {
        "constraints": [c.get("name", c.get("value", "?")) for c in constraints],
        "necessities": [c.get("necessity", "must") for c in constraints],
        "matrix": matrix,
    }


def _build_comparison_table(
    top_papers: List[Dict[str, Any]], query: str
) -> List[Dict[str, Any]]:
    """Generate method comparison table via LLM."""
    if not top_papers:
        return []

    enabled = config.get("organization", "comparison_table", "enabled", default=True)
    if not enabled:
        return []

    client = LLMClient(
        model=config.get("organization", "comparison_table", "model", default="deepseek-v4-flash"),
        provider=config.get("organization", "comparison_table", "model_provider", default="dashscope"),
        temperature=0.0,
        max_tokens=2000,
    )

    paper_data = [
        {"title": p.get("title", ""), "abstract": (p.get("abstract") or "")[:500], "year": p.get("year")}
        for p in top_papers
    ]

    prompt = COMPARISON_TABLE_PROMPT.format(
        query=query,
        paper_list_json=json.dumps(paper_data, ensure_ascii=False, indent=2),
    )

    try:
        return client.chat_json(prompt, system=COMPARISON_TABLE_SYSTEM, repair_json=True)
    except Exception as e:
        logger.warning(f"Comparison table generation failed: {e}")
        return paper_data  # Fallback: raw paper data


def _build_citation_graph_data(
    papers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build citation graph data structure for Mermaid/JSON output."""
    nodes = []
    edges = []

    paper_id_to_idx = {}
    for i, p in enumerate(papers):
        pid = p.get("paper_id", str(i))
        paper_id_to_idx[pid] = i
        nodes.append({
            "id": i,
            "label": (p.get("title") or "")[:60],
            "year": p.get("year"),
            "score": p.get("_final_score"),
        })

    # Build edges based on citation relationships from S2 data
    # (This uses whatever citation data was fetched during retrieval)
    for i, p in enumerate(papers):
        refs = p.get("_references", [])
        for ref_id in refs:
            if ref_id in paper_id_to_idx:
                edges.append({"from": i, "to": paper_id_to_idx[ref_id]})

    return {"nodes": nodes, "edges": edges}


def generate_markdown_output(
    organized_results: List[Dict[str, Any]],
    output_file: str = "output/results.md",
) -> str:
    """
    Generate a readable Markdown report from organized results.
    """
    lines = [
        "# Academic Paper Search Results",
        "",
        f"*Generated on: 2026-08-06*",
        f"*Total queries processed: {len(organized_results)}*",
        "",
        "---",
        "",
    ]

    for i, rr in enumerate(organized_results):
        qid = rr["qid"]
        query = rr["query"]
        output = rr.get("structured_output", {})
        layers = output.get("layers", {})
        matrix = output.get("constraint_matrix", {})
        comp_table = output.get("comparison_table", [])

        lines.extend([
            f"## Query {i+1}: {qid}",
            "",
            f"**Query:** {query}",
            "",
        ])

        # Layers
        for layer_name, layer_papers in layers.items():
            if not layer_papers:
                continue
            label = {"highly_relevant": "Highly Relevant",
                     "partially_relevant": "Partially Relevant",
                     "background": "Background"}.get(layer_name, layer_name)
            lines.append(f"### {label} ({len(layer_papers)} papers)")
            lines.append("")
            lines.append("| # | Title | Year | Venue | Score | Evidence |")
            lines.append("|---|-------|------|-------|-------|----------|")
            for j, p in enumerate(layer_papers[:10]):
                title = (p.get("title") or "")[:60]
                year = str(p.get("year") or "?")
                venue = str(p.get("venue") or "?")[:20]
                score = f"{(p.get('score') or 0):.1f}"
                evidence = (p.get("evidence") or "")[:80]
                lines.append(f"| {j+1} | {title} | {year} | {venue} | {score} | {evidence} |")
            lines.append("")

        # Constraint Matrix
        if matrix.get("constraints"):
            lines.append("### Constraint Satisfaction Matrix")
            lines.append("")
            constraints = matrix["constraints"]
            necessities = matrix.get("necessities", [])
            header = "| Paper | " + " | ".join(f"{c} ({n})" for c, n in zip(constraints, necessities)) + " | Score |"
            lines.append(header)
            lines.append("|" + "|".join(["---"] * (len(constraints) + 2)) + "|")
            for row in matrix.get("matrix", [])[:10]:
                vals = " | ".join(str(row.get(c, "-")) for c in constraints)
                lines.append(f"| {row['title'][:40]} | {vals} | {row['score']:.1f} |")
            lines.append("")

        # Comparison Table
        if comp_table:
            lines.append("### Method Comparison")
            lines.append("")
            if isinstance(comp_table, list) and len(comp_table) > 0:
                keys = list(comp_table[0].keys()) if isinstance(comp_table[0], dict) else []
                if keys:
                    lines.append("| " + " | ".join(keys) + " |")
                    lines.append("|" + "|".join(["---"] * len(keys)) + "|")
                    for row in comp_table[:5]:
                        vals = " | ".join(str(row.get(k, ""))[:40] for k in keys)
                        lines.append(f"| {vals} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Write to file
    import os
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Markdown report saved to: {output_file}")
    return "\n".join(lines)
