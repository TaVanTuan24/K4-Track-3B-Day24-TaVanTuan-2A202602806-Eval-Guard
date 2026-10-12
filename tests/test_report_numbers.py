"""The Markdown reports must agree with the artifacts they quote.

`validate_markdown` only rejects placeholders and empty cells — it cannot tell that a
number in prose was wrong. That is exactly how the blueprint ended up claiming
"multi_hop and adversarial both have avg_score below 0.65" while multi_hop's measured
avg_score is 0.662252 (above the threshold). These tests parse the tables in the
reports and recompute the same values from `artifacts/`, so a report can no longer drift
from the evidence without turning a test red.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import config
from src.experiment_support import read_json

pytestmark = pytest.mark.support

REPORTS = Path(config.ROOT_DIR) / "doc" / "reports"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _row_for(path: Path, label: str) -> list[str]:
    """Cells of the first Markdown table row whose first cell is `label`.

    A trailing qualifier is allowed so a row labelled `factual (20)` matches `factual`.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if cells and (cells[0] == label or cells[0].startswith(label + " ")):
            return [cell.strip("*") for cell in cells]
    raise AssertionError(f"{path.name}: no table row starting with {label!r}")


def _numbers(cells: list[str]) -> list[float]:
    return [float(value) for cell in cells for value in re.findall(r"-?\d+\.\d+", cell)]


@pytest.fixture(scope="module")
def phase_a() -> dict:
    return read_json(Path(config.ARTIFACTS_DIR) / "ragas_50q.json")


def test_failure_clusters_per_distribution_table_matches_artifact(phase_a):
    report = REPORTS / "failure_clusters.md"
    for metric in METRICS + ("avg_score",):
        cells = _row_for(report, metric)
        printed = _numbers(cells)
        assert len(printed) == 3, f"{metric}: expected 3 distribution values, got {cells}"
        expected = [round(phase_a["per_distribution"][d][metric], 4)
                    for d in ("factual", "multi_hop", "adversarial")]
        for got, want in zip(printed, expected):
            assert got == pytest.approx(want, abs=1e-4), f"{metric}: report says {got}, artifact {want}"


def test_global_averages_quoted_in_blueprint_match_artifact(phase_a):
    rows = phase_a["per_question"]

    def mean(metric):
        return sum(r[metric] for r in rows) / len(rows)

    report = REPORTS / "blueprint.md"
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall",
                   "avg_score"):
        cells = _row_for(report, metric)
        printed = _numbers(cells)[0]
        want = mean(metric) if metric != "avg_score" else sum(
            sum(r[m] for m in METRICS) / 4 for r in rows) / len(rows)
        assert printed == pytest.approx(want, abs=1e-6), f"{metric}: report {printed}, artifact {want}"


def test_blueprint_gate_verdicts_are_computed_correctly(phase_a):
    """Every Đạt/Không đạt cell must follow from its own value and the stated threshold."""
    report = REPORTS / "blueprint.md"
    thresholds = {"faithfulness": 0.75, "avg_score": 0.65}
    for distribution in ("factual", "multi_hop", "adversarial"):
        cells = _row_for(report, distribution)
        # | distribution | faith | verdict | avg | verdict |
        faith, faith_verdict, avg, avg_verdict = (
            float(cells[1]), cells[2], float(cells[3]), cells[4])
        assert faith == pytest.approx(phase_a["per_distribution"][distribution]["faithfulness"], abs=1e-4)
        assert avg == pytest.approx(phase_a["per_distribution"][distribution]["avg_score"], abs=1e-4)
        for value, verdict, threshold in ((faith, faith_verdict, thresholds["faithfulness"]),
                                          (avg, avg_verdict, thresholds["avg_score"])):
            expected = "Đạt" if value >= threshold else "Không đạt"
            assert verdict == expected, (
                f"{distribution}: value {value} vs threshold {threshold} should read {expected!r}, "
                f"report says {verdict!r}")


def test_blueprint_does_not_repeat_the_corrected_multi_hop_claim(phase_a):
    """Regression for the wrong sentence: multi_hop's avg_score is ABOVE 0.65."""
    text = (REPORTS / "blueprint.md").read_text(encoding="utf-8")
    assert "đều dưới 0.65" not in text, "the corrected claim reappeared"
    multi_hop_avg = phase_a["per_distribution"]["multi_hop"]["avg_score"]
    assert multi_hop_avg >= 0.65
    adversary_avg = phase_a["per_distribution"]["adversarial"]["avg_score"]
    assert adversary_avg < 0.65
    # Both facts must be visible in the report.
    assert f"{multi_hop_avg:.6f}" in text and f"{adversary_avg:.6f}" in text


def test_blueprint_latency_table_matches_guard_artifact():
    report = REPORTS / "blueprint.md"
    latency = read_json(Path(config.ARTIFACTS_DIR) / "guard_results.json")["latency"]
    for label, key in (("Presidio", "presidio_ms"), ("NeMo input", "nemo_input_ms"),
                       ("Input guard sequential", "input_guard_ms")):
        printed = _numbers(_row_for(report, label))
        assert len(printed) == 3, f"{label}: expected p50/p95/p99"
        for got, percentile in zip(printed, ("p50", "p95", "p99")):
            assert got == pytest.approx(latency[key][percentile], abs=0.01), \
                f"{label}.{percentile}: report {got}, artifact {latency[key][percentile]}"
    # The out-of-scope rows must stay explicitly unmeasured rather than guessed.
    for label in ("RAG", "NeMo output", "Full guard / short-circuit request"):
        assert "chưa đo" in " ".join(_row_for(report, label))


def test_blueprint_budget_verdict_matches_artifact():
    report = REPORTS / "blueprint.md"
    latency = read_json(Path(config.ARTIFACTS_DIR) / "guard_results.json")["latency"]
    assert latency["latency_budget_ok"] is False
    assert f"{latency['input_guard_ms']['p95']:.2f}" in report.read_text(encoding="utf-8")
    assert "False" in report.read_text(encoding="utf-8")


def test_bias_report_numbers_match_judge_artifact():
    report = REPORTS / "bias_report.md"
    judge = read_json(Path(config.ARTIFACTS_DIR) / "judge_results.json")
    bias, calibration = judge["bias_report"], judge["calibration"]
    text = report.read_text(encoding="utf-8")
    assert f"{bias['position_bias_count']} / {bias['position_bias_rate']}" in text
    assert f"{bias['verbosity_bias']}" in text
    assert f"{calibration['cohen_kappa']}" in text
    # Five pairs, ten calibration rows, as required by the brief.
    assert len(judge["pairwise_results"]) == 5
    assert len(judge["binary_judgments"]) == 10
    for row in judge["pairwise_results"]:
        assert f"| {row['question_id']} |" in text, f"pair id {row['question_id']} missing from report"
    for row in judge["binary_judgments"]:
        assert f"| {row['question_id']} |" in text, \
            f"calibration id {row['question_id']} missing from report"


def test_guard_report_attack_breakdown_matches_artifact():
    from collections import Counter

    report = REPORTS / "blueprint.md"
    guard = read_json(Path(config.ARTIFACTS_DIR) / "guard_results.json")
    counts = Counter(row["blocked_by"] for row in guard["adversarial"]["results"])
    text = report.read_text(encoding="utf-8")
    assert guard["adversarial"]["passed"] == 20 and guard["clean"]["passed"] == 8
    assert counts["presidio"] == 4 and counts["nemo_input"] == 16
    assert "20/20" in text and "8/8" in text
