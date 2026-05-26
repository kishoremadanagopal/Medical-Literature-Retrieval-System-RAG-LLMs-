"""Evaluation — metrics, harness, and benchmarks."""

from src.evaluation.metrics import (
    compute_retrieval_metrics,
    compute_generation_metrics,
    classification_report,
    benchmark_latency,
)
from src.evaluation.eval_harness import (
    build_eval_set,
    evaluate_retrieval,
    evaluate_classification,
    benchmark_stores,
)

__all__ = [
    "compute_retrieval_metrics",
    "compute_generation_metrics",
    "classification_report",
    "benchmark_latency",
    "build_eval_set",
    "evaluate_retrieval",
    "evaluate_classification",
    "benchmark_stores",
]
