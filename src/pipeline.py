# ============================================================
# src/pipeline.py - End-to-end pipeline orchestrator
# ============================================================

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.utils.config_loader import load_config, config
from src.utils.logger import setup_logger
from src.utils.cache import ResultCache
from src.llm.token_counter import llm_stats, reset_token_stats
from src.utils.data_loader import (
    load_pasa_dataset, load_asta_dataset,
)
from src.step1_query_understanding import step1_process_all
from src.step2_hybrid_retrieval import step2_retrieve_all
from src.step3_ranking import step3_rank_all
from src.step4_result_organization import step4_organize_all, generate_markdown_output


def run_pipeline(
    config_path: str = "config/config.yaml",
    steps: Optional[List[int]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Main pipeline entry point.

    Args:
        config_path: Path to YAML config file
        steps: List of step numbers to run, e.g. [1,2,3,4]
               If None, uses config setting.
        overrides: Nested path -> value overrides, e.g.
                   {("dataset", "name"): "asta",
                    ("dataset", "query_filter", "max_queries"): 10}

    Returns:
        Final structured results.
    """
    # ---- Setup ----
    cfg = load_config(config_path)
    if overrides:
        _apply_overrides(cfg, overrides)
    logger = setup_logger(cfg)
    steps = steps or cfg.get("pipeline", "steps", default=[1, 2, 3, 4])

    # Reset token stats for this run
    reset_token_stats()

    logger.info(f"Pipeline starting. Steps: {steps}")
    logger.info(f"Dataset: {cfg.get('dataset', 'name', default='pasa')}")

    # Cache
    use_cache = cfg.get("pipeline", "use_cache", default=True)
    cache = ResultCache(cfg.get("pipeline", "cache_dir", default="cache")) if use_cache else None

    # ---- Load Dataset ----
    dataset_items = _load_dataset(cfg)
    logger.info(f"Loaded {len(dataset_items)} queries")

    # ---- Step 1: Query Understanding ----
    if 1 in steps:
        logger.info("=" * 60)
        logger.info("STEP 1: Query Understanding")
        logger.info("=" * 60)

        parsed_queries = step1_process_all(dataset_items)

        if cache:
            cache.set("step1", {"dataset": cfg.get("dataset", "name")}, parsed_queries)

        _save_intermediate(parsed_queries, "step1_parsed_queries.json")
        logger.info(f"Step 1 complete. {len(parsed_queries)} queries parsed.")
    else:
        parsed_queries = _load_or_cache(cache, "step1", dataset_items)

    final_results = parsed_queries
    last_step = max(steps) if steps else 1

    # ---- Step 2: Hybrid Retrieval ----
    if last_step >= 2:
        if 2 in steps:
            logger.info("=" * 60)
            logger.info("STEP 2: Hybrid Retrieval")
            logger.info("=" * 60)

            retrieval_results = step2_retrieve_all(parsed_queries)

            # Attach parsed data for later steps
            for i, rr in enumerate(retrieval_results):
                if i < len(parsed_queries):
                    rr["parsed"] = parsed_queries[i].get("parsed", {})

            if cache:
                cache.set("step2", {"dataset": cfg.get("dataset", "name")}, retrieval_results)

            _save_intermediate(retrieval_results, "step2_retrieval_results.json")
            logger.info(f"Step 2 complete.")
        else:
            retrieval_results = _load_or_cache(cache, "step2", parsed_queries)

        final_results = retrieval_results

    # ---- Step 3: Ranking ----
    if last_step >= 3:
        if 3 in steps:
            logger.info("=" * 60)
            logger.info("STEP 3: Ranking")
            logger.info("=" * 60)

            ranking_results = step3_rank_all(retrieval_results)

            if cache:
                cache.set("step3", {"dataset": cfg.get("dataset", "name")}, ranking_results)

            _save_intermediate(ranking_results, "step3_ranking_results.json")
            logger.info(f"Step 3 complete.")
        else:
            ranking_results = _load_or_cache(cache, "step3", retrieval_results)

        final_results = ranking_results

    # ---- Step 4: Result Organization ----
    if last_step >= 4:
        if 4 in steps:
            logger.info("=" * 60)
            logger.info("STEP 4: Result Organization")
            logger.info("=" * 60)

            final_results = step4_organize_all(ranking_results)

            # Generate Markdown report
            output_file = str(config.project_root / "output" / "results.md")
            generate_markdown_output(final_results, output_file)

            _save_intermediate(final_results, "step4_final_results.json")
            logger.info(f"Step 4 complete. Report: {output_file}")
        else:
            final_results = ranking_results

    # ---- Token Stats ----
    stats = llm_stats.snapshot()
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  LLM calls:     {stats['total_calls']}")
    logger.info(f"  Retries:       {stats['total_retries']}")
    logger.info(f"  Prompt tokens: {stats['total_prompt_tokens']:,}")
    logger.info(f"  Compl tokens:  {stats['total_completion_tokens']:,}")
    logger.info(f"  Total tokens:  {stats['total_tokens']:,}")
    logger.info(f"  Est cost:      ${stats['estimated_cost_usd']:.4f}")
    logger.info("=" * 60)

    return final_results


def _apply_overrides(cfg, overrides: Dict) -> None:
    """Apply nested path -> value overrides to config."""
    for path, value in overrides.items():
        d = cfg._config
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value


def _load_dataset(cfg) -> List[Dict[str, Any]]:
    """Load dataset based on config."""
    ds_config = cfg.dataset
    ds_name = ds_config.get("name", "pasa")
    qf = ds_config.get("query_filter", {})

    max_q = qf.get("max_queries", 0)
    qids = qf.get("query_ids", []) or None

    if ds_name == "pasa":
        return load_pasa_dataset(
            ds_config["pasa_data_path"],
            max_queries=max_q,
            query_ids=qids,
        )
    elif ds_name == "asta":
        types = qf.get("asta_query_types", ["semantic", "specific", "metadata"])
        return load_asta_dataset(
            ds_config["asta_data_path"],
            val_path=ds_config.get("asta_val_data_path"),
            query_types=types,
            max_queries=max_q,
        )
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")


def _save_intermediate(data: Any, filename: str) -> None:
    """Save intermediate results to output directory."""
    out_dir = config.project_root / "output"
    os.makedirs(out_dir, exist_ok=True)
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {path}")


def _load_or_cache(cache, step: str, prev_data) -> Any:
    """Try loading from cache, otherwise raise (step must have been run)."""
    if cache:
        cached = cache.get(step, {"dataset": config.get("dataset", "name")})
        if cached is not None:
            print(f"[Cache] Loaded {step} from cache")
            return cached
    raise RuntimeError(
        f"Step {step} results not found in cache. "
        f"Run with step {step} first, or enable caching in config."
    )
