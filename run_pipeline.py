#!/usr/bin/env python3
# ============================================================
# run_pipeline.py - Main entry point for paper search pipeline
#
# Usage:
#   python run_pipeline.py                          # Run all steps with default config
#   python run_pipeline.py --steps 1 2              # Run only steps 1 and 2
#   python run_pipeline.py --dataset asta           # Use Asta dataset
#   python run_pipeline.py --max-queries 10         # Limit to 10 queries
#   python run_pipeline.py --query-type semantic    # Asta: only semantic queries
#   python run_pipeline.py --no-cache               # Disable result caching
# ============================================================

import sys
import argparse
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline
from src.utils.config_loader import load_config, config
from src.llm.token_counter import llm_stats


def main():
    parser = argparse.ArgumentParser(
        description="Academic Paper Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                                    # Full pipeline, PaSa dataset
  python run_pipeline.py --steps 1 2                        # Only query understanding + retrieval
  python run_pipeline.py --dataset asta --max-queries 10    # Asta dataset, 10 queries
  python run_pipeline.py --dataset asta --query-type semantic --query-type specific
  python run_pipeline.py --no-cache                         # Fresh run, no cache
        """,
    )

    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Pipeline steps to run (1=query, 2=retrieval, 3=ranking, 4=organize)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["pasa", "asta"],
                        help="Dataset to use (overrides config)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Max number of queries to process (0=all)")
    parser.add_argument("--query-type", type=str, action="append", default=None,
                        choices=["semantic", "specific", "metadata"],
                        help="Asta query types to include (can repeat)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable result caching")

    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)

    # --- Build overrides from CLI args ---
    overrides = {}
    if args.dataset:
        overrides[("dataset", "name")] = args.dataset
    if args.max_queries is not None:
        overrides[("dataset", "query_filter", "max_queries")] = args.max_queries
    if args.query_type:
        overrides[("dataset", "query_filter", "asta_query_types")] = args.query_type
    if args.no_cache:
        overrides[("pipeline", "use_cache")] = False

    # --- Print run summary ---
    print("=" * 60)
    print("  Paper Search Pipeline")
    print("=" * 60)
    print(f"  Dataset:      {args.dataset or cfg.get('dataset', 'name')}")
    print(f"  Max queries:  {args.max_queries if args.max_queries is not None else (cfg.get('dataset', 'query_filter', 'max_queries', default=0) or 'all')}")
    print(f"  Steps:        {args.steps or cfg.get('pipeline', 'steps')}")
    print(f"  Cache:        {'OFF' if args.no_cache else 'ON'}")
    print(f"  LLM Model:    {cfg.get('query_understanding', 'model')}")
    print(f"  Sources:      {cfg.get('retrieval', 'sources')}")
    print("=" * 60)
    print()

    # --- Run ---
    results = run_pipeline(config_path=args.config, steps=args.steps, overrides=overrides or None)

    # --- Summary ---
    stats = llm_stats.snapshot()
    print()
    print("=" * 60)
    print(f"  Pipeline Complete — {len(results)} queries processed")
    print(f"  LLM calls: {stats['total_calls']} | Retries: {stats['total_retries']}")
    print(f"  Tokens: {stats['total_tokens']:,} | Est cost: ${stats['estimated_cost_usd']:.4f}")
    print(f"  Output: output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
