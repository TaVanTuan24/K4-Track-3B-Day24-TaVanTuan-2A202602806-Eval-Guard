from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH, ARTIFACTS_DIR
from src.experiment_support import write_json

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}

DISTRIBUTIONS = ("factual", "multi_hop", "adversarial")
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# ─── Root-cause vocabulary (Task 3 diagnosis) ────────────────────────────────
# `DIAGNOSTIC_TREE` maps the metric that scored lowest to a generic cause. That is a
# useful default, but it is wrong for the most common failure in this corpus: an answer
# of "Không tìm thấy." scores faithfulness 0.0 without containing a single unsupported
# claim, so calling it "LLM hallucinating" names the opposite problem. The classifier
# below refines the label with evidence that is actually in the row (answer text,
# retrieved contexts, reference) and falls back to DIAGNOSTIC_TREE otherwise.
#
# Contract kept intact: `worst_metric` is still the argmin of the four metrics, the
# bottom-10 membership and ordering are unchanged, and metrics are never rewritten.
ABSTENTION_MARKERS = (
    "không tìm thấy", "không có thông tin", "không đề cập", "chưa có thông tin",
    "không đủ thông tin", "không thể trả lời", "không cung cấp thông tin",
)
# A version is "obsolete" when the reference says so in the same sentence.
OBSOLETE_MARKERS = ("hết hiệu lực", "đã bị thay thế", "thay thế hoàn toàn", "không còn hiệu lực", "bản cũ")

# NOTE on polarity (negation) errors: they are NOT auto-classified here. A lexical rule
# ("reference forbids + answer lacks a negation word") flags a correct answer as an
# error — measured on this run it mislabelled row 45, whose answer ("Có, MFA bắt buộc
# cho tất cả nhân viên…") matches the reference with faithfulness 1.0. Polarity is
# therefore verified by reading the row in the report instead of by a keyword test.
_VERSION_RE = re.compile(r"\bv(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d[\d.,]*\d|\d")
_SENTENCE_SPLIT = re.compile(r"[.;:\n]+")


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

@dataclass
class RagasEvaluation:
    """Task 2 output: measured rows plus model identifiers reported by m4."""
    results: list[RagasResult]
    evaluator_models: list[str]

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python scripts/setup_answers.py --live"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict]) -> dict:
    """Provided exporter: preserve the student-produced Task 3 rows verbatim."""
    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_question": [asdict(r) for r in results],
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom,
    }
    return report


def save_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict],
                        path: str = os.path.join(ARTIFACTS_DIR, "ragas_50q.json")) -> None:
    write_json(path, build_phase_a_report(results, clusters, bottom))



# ─── Tasks 1-4 ───────────────────────────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups: dict[str, list[dict]] = {name: [] for name in DISTRIBUTIONS}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(f"Unknown distribution {distribution!r} for id {item.get('id')}.")
        groups[distribution].append(item)
    return groups


def _metric_value(record, name: str) -> float:
    """Read one metric from an m4 row, accepting an object or a plain mapping."""
    value = record.get(name) if isinstance(record, dict) else getattr(record, name, None)
    if value is None:
        raise ValueError(f"m4 per_question row is missing metric {name!r}.")
    number = float(value)
    if number != number:
        raise ValueError(f"m4 returned NaN for {name}; refusing to invent a score.")
    return number


def run_ragas_50q(answers: list[dict]) -> RagasEvaluation:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi.

    Calls the Day 18 evaluator `src.m4_eval.evaluate_ragas` and keeps its reported
    evaluator models, so metadata names the models that really ran.
    """
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:
        print("⚠️  Không tìm thấy src/m4_eval.py — đã copy từ Day 18 chưa?")
        raise ValueError("Missing m4 evaluator; cannot declare its actual models")

    questions     = [a["question"]     for a in answers]
    ans_texts     = [a["answer"]       for a in answers]
    contexts      = [a["contexts"]     for a in answers]
    ground_truths = [a["ground_truth"] for a in answers]

    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    if not isinstance(raw, dict) or not isinstance(raw.get("per_question"), list):
        raise ValueError("m4.evaluate_ragas must return a dict with a 'per_question' list.")
    per_q = raw["per_question"]
    if len(per_q) != len(answers):
        raise ValueError(f"m4 returned {len(per_q)} rows for {len(answers)} answers.")

    results = [
        RagasResult(
            question_id=a["id"], distribution=a["distribution"],
            question=a["question"], answer=a["answer"],
            contexts=list(a["contexts"]), ground_truth=a["ground_truth"],
            faithfulness=_metric_value(pq, "faithfulness"),
            answer_relevancy=_metric_value(pq, "answer_relevancy"),
            context_precision=_metric_value(pq, "context_precision"),
            context_recall=_metric_value(pq, "context_recall"),
        )
        for a, pq in zip(answers, per_q)
    ]
    metadata = raw.get("metadata") or {}
    evaluator_models = metadata.get("evaluator_models")
    if not isinstance(evaluator_models, list) or not evaluator_models:
        raise ValueError("m4 metadata.evaluator_models must list the models actually used.")
    return RagasEvaluation(results, evaluator_models)


def is_abstention(answer: str) -> bool:
    """The model declined instead of answering — the opposite of hallucinating."""
    text = (answer or "").strip().lower().rstrip(".")
    return any(text.startswith(marker) or text == marker for marker in ABSTENTION_MARKERS)


def reference_numbers(text: str) -> set[str]:
    """Digit strings with separators removed, keeping only distinctive values (2+ digits).

    Comparing numbers is the least ambiguous "is the needed fact in the context" test:
    amounts, day counts, extension numbers and thresholds are what the answers turn on.
    Version labels are removed first, so `v1.3` cannot masquerade as the number 13.
    """
    numbers = set()
    for raw in _NUMBER_RE.findall(_VERSION_RE.sub(" ", text or "")):
        digits = re.sub(r"[.,]", "", raw)
        if len(digits) >= 2:
            numbers.add(digits)
    return numbers


def version_markers(text: str) -> set[str]:
    """Version labels such as v1.0 / v2024 / v1.3 (normalised without the `v`)."""
    return {match.group(1) for match in _VERSION_RE.finditer(text or "")}


def obsolete_versions(reference: str) -> set[str]:
    """Versions the reference itself marks as superseded, judged sentence by sentence."""
    obsolete: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(reference or ""):
        stripped = sentence.strip().lower()
        if any(marker in stripped for marker in OBSOLETE_MARKERS):
            obsolete |= version_markers(sentence)
    return obsolete


def current_versions(reference: str) -> set[str]:
    """Versions the reference treats as in force = all versions minus obsolete ones."""
    return version_markers(reference) - obsolete_versions(reference)


def is_version_conflict(result: RagasResult) -> bool:
    """True only when the evidence proves an obsolete-only retrieval.

    Requires: the reference distinguishes superseded from in-force versions, the
    contexts surface an obsolete version, and no in-force version is present at all.
    A row whose contexts do contain the in-force version is NOT called a conflict —
    that keeps the label tied to bytes instead of to a guess.
    """
    obsolete = obsolete_versions(result.ground_truth)
    current = current_versions(result.ground_truth)
    if not obsolete or not current:
        return False
    context_versions: set[str] = set()
    for context in result.contexts:
        context_versions |= version_markers(context)
    return bool(context_versions & obsolete) and not (context_versions & current)


def missing_reference_numbers(result: RagasResult) -> set[str]:
    """Distinctive values of the reference that never appear in any retrieved context."""
    wanted = reference_numbers(result.ground_truth)
    present = reference_numbers(" ".join(result.contexts))
    return wanted - present


def diagnose(result: RagasResult) -> tuple[str, str]:
    """Evidence-based root cause for one row, with `DIAGNOSTIC_TREE` as the fallback.

    Order matters: an obsolete-only retrieval is specific and checkable, so it is tested
    before the generic abstention split; the metric-based default only runs when nothing
    more precise applies. Polarity (negation) errors are deliberately NOT auto-classified
    — see the note next to the vocabulary constants above.
    """
    missing = missing_reference_numbers(result)

    if is_version_conflict(result):
        return ("Version conflict — context chỉ chứa bản đã hết hiệu lực, thiếu bản hiện hành",
                "Lọc theo phiên bản (metadata filter trên auto_metadata.version) và nêu rõ "
                "bản hiện hành so với bản đã bị thay thế trong prompt generation.")

    if is_abstention(result.answer):
        if missing:
            return (f"Retrieval failure + abstention — model từ chối và context thiếu "
                    f"{len(missing)} dữ kiện của reference (ví dụ {sorted(missing)[:3]})",
                    "Tăng recall: cải thiện chunking/top-k, thêm BM25 cho số và mã, và chỉ "
                    "cho phép trả lời 'Không tìm thấy.' khi mọi context đều không liên quan.")
        return ("Abstention — context có bằng chứng nhưng model vẫn từ chối trả lời",
                "Nới điều kiện trả lời 'Không tìm thấy.' trong system prompt, yêu cầu dùng "
                "mọi context liên quan và hạ temperature; đây là lỗi generation, không phải hallucination.")

    return DIAGNOSTIC_TREE[result.worst_metric]


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy 10 câu hỏi có avg_score thấp nhất.

    Ranking and `worst_metric` come straight from the measured metrics; only the
    diagnosis/fix text is refined by `diagnose()`.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    ordered = sorted(results, key=lambda r: r.avg_score)[:10]
    rows: list[dict] = []
    for rank, result in enumerate(ordered, start=1):
        diagnosis, fix = diagnose(result)
        rows.append({
            "rank": rank,
            "question_id": result.question_id,
            "distribution": result.distribution,
            "question": result.question,
            "avg_score": round(result.avg_score, 4),
            "worst_metric": result.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return rows


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Matrix thống kê worst_metric của mọi câu, không coi mọi câu là một failure.
    Nhóm yếu nhất = distribution có avg_score trung bình thấp nhất (bỏ nhóm rỗng).
    Tổng matrix theo cột chỉ là kích thước nhóm 20/20/10, không đo chất lượng.
    """
    matrix = {metric: {dist: 0 for dist in DISTRIBUTIONS} for metric in METRICS}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        matrix[result.worst_metric][result.distribution] += 1
        totals[result.distribution] = totals.get(result.distribution, 0.0) + result.avg_score
        counts[result.distribution] = counts.get(result.distribution, 0) + 1

    # Empty groups are dropped from the division, never scored as zero quality.
    means = {dist: totals[dist] / counts[dist] for dist in DISTRIBUTIONS if counts.get(dist)}
    if not means:
        raise ValueError("cluster_analysis needs at least one scored question.")
    weakest_distribution = min(means, key=means.get)
    metric_totals = {metric: sum(matrix[metric].values()) for metric in METRICS}
    dominant_metric = max(metric_totals, key=metric_totals.get)
    insight = (
        f"Distribution '{weakest_distribution}' có avg_score trung bình thấp nhất "
        f"({means[weakest_distribution]:.3f}). Metric yếu chủ đạo toàn bộ {len(results)} câu là "
        f"'{dominant_metric}' ({metric_totals[dominant_metric]} câu có metric này thấp nhất) — "
        f"{DIAGNOSTIC_TREE[dominant_metric][0]}. "
        f"Gợi ý: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    )
    return {
        "matrix": matrix,
        # The schema wants all three distributions when this block is present, so it
        # is only emitted for a complete 20/20/10 run instead of inventing a mean
        # for an empty group.
        **({"distribution_avg_scores": {dist: round(means[dist], 6) for dist in DISTRIBUTIONS}}
           if all(dist in means for dist in DISTRIBUTIONS) else {}),
        "dominant_failure_distribution": weakest_distribution,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "a", *sys.argv[1:]]))
