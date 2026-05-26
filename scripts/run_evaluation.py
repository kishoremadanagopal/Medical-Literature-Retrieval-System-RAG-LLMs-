#!/usr/bin/env python3
"""
Run the evaluation suite (Step 6).

Usage:
    python scripts/run_evaluation.py                  # Full eval suite
    python scripts/run_evaluation.py --retrieval-only  # Just retrieval metrics
    python scripts/run_evaluation.py --latency-only    # Just latency benchmark
    python scripts/run_evaluation.py --classify-only   # Just classification eval
    python scripts/run_evaluation.py --n-queries 50    # Smaller eval set (faster)
    python scripts/run_evaluation.py --output results.json  # Save to JSON

Requires: Steps 2-4 completed (indices built).
Produces: The metrics claimed in the README.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import settings
from config.logging_config import get_logger

logger = get_logger("run_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--latency-only", action="store_true")
    parser.add_argument("--classify-only", action="store_true")
    parser.add_argument("--n-queries", type=int, default=200, help="Eval set size")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None, help="Save results to JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings.ensure_directories()
    run_all = not (args.retrieval_only or args.latency_only or args.classify_only)

    chunks_path = settings.processed_data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"\n  ✗ Chunks not found at {chunks_path}")
        print("    Run Steps 2-3 first.")
        return 1

    print("\n🧪 Medical RAG — Evaluation Suite\n")

    # ── Initialize stores ───────────────────────────────────────────
    from src.processing.embedder import BioBERTEmbedder
    from src.retrieval.chroma_store import ChromaStore
    from src.retrieval.faiss_store import FAISSStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.evaluation.eval_harness import (
        benchmark_stores,
        build_eval_set,
        compare_retrieval_methods,
        evaluate_classification,
    )

    results = {}

    print("  Loading components...", flush=True)
    embedder = BioBERTEmbedder()

    chroma = ChromaStore()
    faiss_store = FAISSStore()
    try:
        faiss_store.load()
    except FileNotFoundError:
        print("  ✗ FAISS index not found. Run Step 4 first.")
        return 1

    retriever = HybridRetriever(chroma, faiss_store)
    print(f"  ✓ Loaded ({chroma.count:,} Chroma / {faiss_store.count:,} FAISS vectors)\n")

    # ── Build eval set ──────────────────────────────────────────────
    if run_all or args.retrieval_only or args.classify_only:
        print(f"  Building eval set ({args.n_queries} queries)...", flush=True)
        eval_set = build_eval_set(chunks_path, n_queries=args.n_queries)
        print(f"  ✓ {len(eval_set)} eval queries\n")

    # ── 1. Retrieval Evaluation ─────────────────────────────────────
    if run_all or args.retrieval_only:
        print("=" * 55)
        print("  📊 RETRIEVAL EVALUATION")
        print("=" * 55)

        start = time.time()
        retrieval_results = compare_retrieval_methods(
            chroma, faiss_store, retriever, embedder, eval_set, top_k=args.top_k,
        )
        elapsed = time.time() - start

        r = retrieval_results
        print(f"\n  FAISS-only:")
        print(f"    NDCG@{args.top_k}:     {r['faiss'][f'mean_ndcg@{args.top_k}']:.4f}")
        print(f"    Precision@{args.top_k}: {r['faiss'][f'mean_precision@{args.top_k}']:.4f}")
        print(f"    MRR:          {r['faiss']['mean_mrr']:.4f}")

        print(f"\n  Ensemble (FAISS + Chroma + RRF):")
        print(f"    NDCG@{args.top_k}:     {r['ensemble'][f'mean_ndcg@{args.top_k}']:.4f}")
        print(f"    Precision@{args.top_k}: {r['ensemble'][f'mean_precision@{args.top_k}']:.4f}")
        print(f"    MRR:          {r['ensemble']['mean_mrr']:.4f}")

        print(f"\n  → Relevance improvement: {r['comparison']['ndcg_improvement_pct']:+.1f}%")
        print(f"  Completed in {elapsed:.1f}s\n")

        results["retrieval"] = retrieval_results

    # ── 2. Classification Evaluation ────────────────────────────────
    if run_all or args.classify_only:
        print("=" * 55)
        print("  🏷️  CLASSIFICATION EVALUATION")
        print("=" * 55)

        start = time.time()
        class_results = evaluate_classification(
            eval_set, retriever, embedder, top_k=args.top_k,
        )
        elapsed = time.time() - start

        accuracy = class_results.get("accuracy", 0)
        macro_f1 = class_results.get("macro_f1", 0)
        print(f"\n  Accuracy:  {accuracy:.1%}")
        print(f"  Macro F1:  {macro_f1:.4f}")
        print(f"  Evaluated: {class_results['evaluated']} samples")
        print(f"  Skipped:   {class_results['skipped_other']} ('other' domain)")

        if "per_class" in class_results:
            print(f"\n  Per-class breakdown:")
            for label, metrics in sorted(class_results["per_class"].items()):
                if metrics.get("support", 0) >= 3:
                    print(f"    {label:20s}  F1={metrics['f1']:.3f}  (n={metrics['support']})")

        print(f"\n  Completed in {elapsed:.1f}s\n")
        results["classification"] = class_results

    # ── 3. Latency Benchmark ────────────────────────────────────────
    if run_all or args.latency_only:
        print("=" * 55)
        print("  ⏱️  LATENCY BENCHMARK")
        print("=" * 55)

        start = time.time()
        latency_results = benchmark_stores(
            chroma, faiss_store, n_queries=100, top_k=args.top_k,
        )
        elapsed = time.time() - start

        f = latency_results["faiss"]
        c = latency_results["chroma"]
        print(f"\n  FAISS ({settings.faiss_index_type}):")
        print(f"    Mean: {f['mean_ms']:.3f} ms  |  p95: {f['p95_ms']:.3f} ms")

        print(f"\n  Chroma:")
        print(f"    Mean: {c['mean_ms']:.3f} ms  |  p95: {c['p95_ms']:.3f} ms")

        print(f"\n  → FAISS is {latency_results['speedup_factor']:.1f}x faster "
              f"({latency_results['speedup_pct']:+.0f}%)")
        print(f"  Completed in {elapsed:.1f}s\n")

        results["latency"] = latency_results

    # ── Summary ─────────────────────────────────────────────────────
    print("=" * 55)
    print("  📋 SUMMARY — README METRICS")
    print("=" * 55)

    if "classification" in results:
        print(f"  Classification accuracy: {results['classification'].get('accuracy', 0):.0%}")
    if "retrieval" in results:
        print(f"  Query relevance lift:    {results['retrieval']['comparison']['ndcg_improvement_pct']:+.1f}%")
    if "latency" in results:
        print(f"  FAISS speedup:           {results['latency']['speedup_pct']:+.0f}%")
    print()

    # ── Save results ────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to {args.output}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
