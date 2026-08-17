#!/usr/bin/env python3
# ============================================================
# run_evaluation.py - Evaluation runner against ground truth
#
# Usage:
#   python run_evaluation.py                                    # Evaluate latest pipeline results
#   python run_evaluation.py --results output/step3_ranking_results.json
#   python run_evaluation.py --dataset both                     # Evaluate on both datasets
# ============================================================

import sys
import os
import json
import argparse
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import load_config
from src.utils.data_loader import (
    load_pasa_dataset, load_asta_dataset, load_asta_normalizer,
    get_query_id, get_query_type, get_gold_answers,
)
from src.evaluation.metrics import evaluate_pasa, evaluate_asta_semantic, evaluate_asta_exact


def main():
    parser = argparse.ArgumentParser(description="Evaluate paper search results")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to pipeline results JSON (default: output/step4_final_results.json)")
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["pasa", "asta", "both"])
    parser.add_argument("--output", type=str, default="output/evaluation_report.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load results
    results_path = args.results or "output/step4_final_results.json"
    if not os.path.exists(results_path):
        # Try step3 results
        results_path = "output/step3_ranking_results.json"
    if not os.path.exists(results_path):
        print(f"[ERROR] No results found. Run run_pipeline.py first.")
        print(f"        Tried: output/step4_final_results.json")
        print(f"        Tried: output/step3_ranking_results.json")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} result entries from {results_path}")

    # Build result index by qid
    results_by_qid = {}
    for r in results:
        qid = r.get("qid", "")
        papers = r.get("ranked_papers", r.get("candidates", []))
        results_by_qid[qid] = papers

    # Evaluate
    all_metrics = {}

    if args.dataset in ("pasa", "both"):
        print("\n=== PaSa (RealScholarQuery) Evaluation ===")
        pasa_metrics = _eval_pasa(cfg, results_by_qid)
        all_metrics["pasa"] = pasa_metrics
        _print_summary(pasa_metrics)

    if args.dataset in ("asta", "both"):
        print("\n=== Asta (PaperFindingBench) Evaluation ===")
        asta_metrics = _eval_asta(cfg, results_by_qid)
        all_metrics["asta"] = asta_metrics
        _print_summary(asta_metrics)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nEvaluation report saved to: {args.output}")


def _eval_pasa(cfg, results_by_qid: Dict[str, List]) -> Dict[str, Any]:
    """Evaluate against PaSa RealScholarQuery."""
    ds_config = cfg.dataset
    queries = load_pasa_dataset(ds_config["pasa_data_path"])

    per_query = []
    for item in queries:
        qid = item["qid"]
        papers = results_by_qid.get(qid, [])
        gold = get_gold_answers(item)

        metrics = evaluate_pasa(papers, gold)
        per_query.append({"qid": qid, **metrics})

    # Aggregate
    return _aggregate(per_query)


def _eval_asta(cfg, results_by_qid: Dict[str, List]) -> Dict[str, Any]:
    """Evaluate against Asta PaperFindingBench."""
    ds_config = cfg.dataset
    queries = load_asta_dataset(
        ds_config["asta_data_path"],
        val_path=ds_config.get("asta_val_data_path"),
    )
    normalizer = load_asta_normalizer(ds_config.get("asta_normalizer_path", ""))

    per_query = []
    for item in queries:
        qid = item["input"]["query_id"]
        papers = results_by_qid.get(qid, [])
        qtype = get_query_type(item)
        gold = get_gold_answers(item)

        if qtype == "semantic":
            norm_val = normalizer.get(qid)
            metrics = evaluate_asta_semantic(papers, gold, norm_val)
        else:
            metrics = evaluate_asta_exact(papers, gold)

        per_query.append({"qid": qid, "type": qtype, **metrics})

    # Aggregate by type
    by_type = {}
    for pq in per_query:
        t = pq.get("type", "semantic")
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(pq)

    result = {}
    for t, items in by_type.items():
        agg = _aggregate(items)
        agg["count"] = len(items)
        result[t] = agg

    result["overall"] = _aggregate(per_query)
    result["overall"]["count"] = len(per_query)
    result["per_query"] = per_query

    return result


def _aggregate(per_query: List[Dict]) -> Dict[str, float]:
    """Compute average metrics across queries."""
    if not per_query:
        return {}

    # Collect all metric keys (skip qid)
    keys = set()
    for pq in per_query:
        keys.update(k for k in pq if k != "qid" and k != "type" and isinstance(pq[k], (int, float)))

    agg = {}
    for key in keys:
        values = [pq.get(key, 0.0) for pq in per_query]
        agg[f"avg_{key}"] = round(sum(values) / len(values), 4)
        agg[f"max_{key}"] = round(max(values), 4)
        agg[f"min_{key}"] = round(min(values), 4)

    return agg


def _print_summary(metrics: Dict) -> None:
    """Print a readable summary."""
    for section, data in metrics.items():
        if section == "per_query":
            continue
        if isinstance(data, dict):
            print(f"\n  [{section}]")
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
