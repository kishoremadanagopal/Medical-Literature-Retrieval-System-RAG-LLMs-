"""
Tests for src/evaluation/ — all metrics computed on known inputs.

Run: pytest tests/test_evaluation.py -v
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import (
    benchmark_latency,
    citation_precision,
    citation_recall,
    classification_accuracy,
    classification_report,
    compute_generation_metrics,
    compute_retrieval_metrics,
    faithfulness_score,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.evaluation.eval_harness import classify_article


# ── Retrieval Metrics ───────────────────────────────────────────────

class TestRetrievalMetrics:
    # Scenario: retrieved [A, B, C, D, E], relevant = {A, C, F}
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}

    def test_precision_at_5(self):
        # 2 relevant out of 5 retrieved
        assert precision_at_k(self.retrieved, self.relevant, 5) == 0.4

    def test_precision_at_2(self):
        # Only A is relevant in top-2 [A, B]
        assert precision_at_k(self.retrieved, self.relevant, 2) == 0.5

    def test_recall_at_5(self):
        # 2 out of 3 relevant docs found
        assert abs(recall_at_k(self.retrieved, self.relevant, 5) - 2 / 3) < 0.001

    def test_recall_at_1(self):
        # Only A found out of {A, C, F}
        assert abs(recall_at_k(self.retrieved, self.relevant, 1) - 1 / 3) < 0.001

    def test_ndcg_perfect_ranking(self):
        # Perfect: all relevant docs at the top
        perfect = ["A", "C", "F", "X", "Y"]
        assert ndcg_at_k(perfect, self.relevant, 5) == 1.0

    def test_ndcg_worst_ranking(self):
        # No relevant docs at all
        assert ndcg_at_k(["X", "Y", "Z"], self.relevant, 3) == 0.0

    def test_ndcg_partial(self):
        # Some relevant docs but not perfectly ranked
        score = ndcg_at_k(self.retrieved, self.relevant, 5)
        assert 0 < score < 1

    def test_mrr_first_is_relevant(self):
        assert mrr(["A", "B", "C"], {"A"}) == 1.0

    def test_mrr_second_is_relevant(self):
        assert mrr(["B", "A", "C"], {"A"}) == 0.5

    def test_mrr_none_relevant(self):
        assert mrr(["X", "Y", "Z"], {"A"}) == 0.0

    def test_compute_retrieval_metrics(self):
        result = compute_retrieval_metrics(self.retrieved, self.relevant, k=5)
        assert "precision@5" in result
        assert "recall@5" in result
        assert "ndcg@5" in result
        assert "mrr" in result
        assert all(0 <= v <= 1 for v in result.values())

    def test_empty_inputs(self):
        assert precision_at_k([], {"A"}, 5) == 0.0
        assert recall_at_k(["A"], set(), 5) == 0.0
        assert ndcg_at_k([], set(), 5) == 0.0


# ── Generation Metrics ──────────────────────────────────────────────

class TestGenerationMetrics:
    def test_citation_precision_all_relevant(self):
        assert citation_precision({"A", "B"}, {"A", "B", "C"}) == 1.0

    def test_citation_precision_half_relevant(self):
        assert citation_precision({"A", "B"}, {"A"}) == 0.5

    def test_citation_precision_none_cited(self):
        assert citation_precision(set(), {"A"}) == 0.0

    def test_citation_recall_all_cited(self):
        assert citation_recall({"A", "B"}, {"A", "B"}) == 1.0

    def test_citation_recall_partial(self):
        assert citation_recall({"A"}, {"A", "B"}) == 0.5

    def test_faithfulness_all_grounded(self):
        assert faithfulness_score({"A", "B"}, {"A", "B", "C"}) == 1.0

    def test_faithfulness_hallucinated(self):
        # Cited "X" which wasn't in retrieved set
        assert faithfulness_score({"A", "X"}, {"A", "B"}) == 0.5

    def test_faithfulness_no_citations(self):
        assert faithfulness_score(set(), {"A"}) == 1.0

    def test_compute_generation_metrics(self):
        result = compute_generation_metrics(
            cited_pmids={"A", "B"},
            relevant_pmids={"A", "C"},
            retrieved_pmids={"A", "B", "C"},
        )
        assert "citation_precision" in result
        assert "citation_recall" in result
        assert "faithfulness" in result


# ── Classification Metrics ──────────────────────────────────────────

class TestClassificationMetrics:
    def test_accuracy_perfect(self):
        assert classification_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_accuracy_half(self):
        assert classification_accuracy(["a", "b"], ["a", "x"]) == 0.5

    def test_accuracy_empty(self):
        assert classification_accuracy([], []) == 0.0

    def test_classify_article_oncology(self):
        assert classify_article(["Neoplasms", "Something Else"]) == "oncology"

    def test_classify_article_genetics(self):
        assert classify_article(["Genomics"]) == "genetics"

    def test_classify_article_other(self):
        assert classify_article(["Unknown Term"]) == "other"

    def test_classify_article_empty(self):
        assert classify_article([]) == "other"

    def test_classification_report_structure(self):
        y_true = ["oncology", "oncology", "genetics", "genetics"]
        y_pred = ["oncology", "genetics", "genetics", "genetics"]
        report = classification_report(y_true, y_pred)
        assert "accuracy" in report
        assert report["accuracy"] == 0.75


# ── Latency Benchmark ───────────────────────────────────────────────

class TestBenchmarkLatency:
    def test_benchmark_returns_metrics(self):
        def fake_search(q):
            time.sleep(0.001)  # 1ms

        result = benchmark_latency(fake_search, ["q1", "q2", "q3"], runs=2)
        assert "p50_ms" in result
        assert "p95_ms" in result
        assert "mean_ms" in result
        assert result["total_queries"] == 6  # 3 queries × 2 runs
        assert result["mean_ms"] > 0
