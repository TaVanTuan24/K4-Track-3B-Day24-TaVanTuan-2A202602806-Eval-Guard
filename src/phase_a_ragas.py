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
# Split on ';', newlines, and a period FOLLOWED BY SPACE — never on a bare '.', because
# that would cut version labels in half ("v2.0 HIỆN HÀNH" became "v2" + "0 ..."), which
# silently emptied `obsolete_versions` and hid the version trap on row 44.
_SENTENCE_SPLIT = re.compile(r"(?:[;\n]+|\.\s+)")
# m5 enrichment prefixes each context with "[topic | doc_type | version]". That header is
# the RETRIEVED DOCUMENT's own version (structured metadata), which is much stronger
# evidence than a version-looking string anywhere in the body: an old document can mention
# a newer version (measured on row 44, whose v1.0 context says "ĐÃ THAY THẾ bởi v2.0").
_CONTEXT_HEADER_RE = re.compile(r"^\s*\[(?P<tag>[^\]]*)\]")
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)

# Coverage floor for "the retrieved set barely overlaps the reference topic". Measured on
# this run's abstention rows: the off-topic retrieval (row 40, contexts about probation
# and data classification while the reference is about incident reporting) scores 0.15,
# while rows whose contexts do contain the needed policy score 0.39-0.85. 0.30 sits in
# that gap; the value is a documented heuristic, not a threshold tuned to pass a test.
COVERAGE_FLOOR = 0.30
# An answer counts as a PARTIAL answer when substantial content follows the abstention
# phrase (row 33 answers the leave-days half and abstains only on the allowance).
PARTIAL_ANSWER_MIN_CHARS = 120

# Very common Vietnamese words carry no topical signal, so they are excluded from the
# coverage measure: including them would inflate coverage for every row equally.
COVERAGE_STOPWORDS = frozenset("""
và của cho các có theo khi là trong một với được hoặc từ này những nhân viên không phải
nên gồm về đến trên dưới sau trước mỗi tất cả như để nếu thì bị do tại bao nhiêu ngày
tháng năm nào gì sao ai đâu cần có thể sẽ đã đang rồi mà hay vài một số
""".split())


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
    """True only when the RETRIEVED DOCUMENT versions prove an obsolete-only retrieval.

    Version identity comes from each context's m5 header (`[topic | doc_type | version]`),
    not from scanning the body: a superseded document can name its replacement inside its
    own text ("ĐÃ THAY THẾ bởi v2.0"), and treating such a mention as "the current version
    was retrieved" would hide a real version trap. Measured on row 44, whose only versioned
    context is the v1.0 document and whose `v2.0` occurrence is exactly such a mention.

    Requires: the reference distinguishes superseded from in-force versions, an obsolete
    document version was retrieved, and no in-force document version was retrieved.
    """
    obsolete = obsolete_versions(result.ground_truth)
    current = current_versions(result.ground_truth)
    if not obsolete or not current:
        return False
    retrieved = retrieved_document_versions(result)
    return bool(retrieved & obsolete) and not (retrieved & current)


def context_header_tag(context: str) -> str | None:
    """The `[topic | doc_type | version]` tag m5 prefixed to a context, if present."""
    match = _CONTEXT_HEADER_RE.match(context or "")
    return match.group("tag").strip() if match else None


def context_document_version(context: str) -> str | None:
    """The version of the document a context came from, read from its own header tag."""
    tag = context_header_tag(context)
    if not tag:
        return None
    parts = [part.strip() for part in tag.split("|")]
    if len(parts) < 3 or not parts[2]:
        return None
    versions = version_markers(parts[2]) or {part.removeprefix("v") for part in (parts[2],)}
    return sorted(versions)[0] if versions else None


def retrieved_document_versions(result: RagasResult) -> set[str]:
    """Versions of the documents actually retrieved (header-based; may be empty)."""
    found = set()
    for context in result.contexts:
        version = context_document_version(context)
        if version:
            found.add(version)
    return found


def version_mentions_beyond_headers(result: RagasResult) -> set[str]:
    """Versions that appear in the bodies but not in any retrieved document's header.

    These are the misleading ones: named by an older document rather than retrieved.
    Reported so a version-conflict diagnosis can say WHY the body text is not evidence.
    """
    in_bodies: set[str] = set()
    for context in result.contexts:
        body = _CONTEXT_HEADER_RE.sub("", context or "")
        in_bodies |= version_markers(body)
    return in_bodies - retrieved_document_versions(result)


def missing_reference_numbers(result: RagasResult) -> set[str]:
    """Distinctive values of the reference that never appear in any retrieved context."""
    wanted = reference_numbers(result.ground_truth)
    present = reference_numbers(" ".join(result.contexts))
    return wanted - present


def content_tokens(text: str) -> set[str]:
    """Topical tokens: length >= 3 and not in the common-word list."""
    return {token for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text or ""))
            if len(token) >= 3 and token not in COVERAGE_STOPWORDS}


def reference_coverage(result: RagasResult) -> float:
    """Fraction of the reference's topical vocabulary that appears in the contexts.

    A cheap, transparent proxy for "did retrieval bring evidence on this topic at all".
    It is only one signal: the diagnosis also checks distinctive numbers and document
    versions, and the Markdown analysis reads the rows.
    """
    wanted = content_tokens(result.ground_truth)
    if not wanted:
        return 1.0
    present = content_tokens(" ".join(result.contexts))
    return len(wanted & present) / len(wanted)


def abstention_kind(answer: str) -> str:
    """`none` | `partial` | `complete`.

    Only an answer that *opens* with an abstention statement counts. A marker buried
    inside a substantive answer is not an abstention — measured on row 39, whose
    comparison table legitimately contains a cell like "không đề cập" for a criterion the
    v1.0 policy never covered; treating that as a partial abstention would have replaced a
    correct diagnosis ("Too many irrelevant chunks") with a wrong one.

    A PARTIAL answer keeps substantive content after the opening phrase. Row 33 is the
    measured example: it correctly answers the leave-days half ("15 + 4 = 19 ngày phép")
    and abstains only on the allowance, so calling it a total abstention would hide that
    half the question was answered and would misdirect the fix.
    """
    text = (answer or "").strip().lstrip("*#-> \t")
    lowered = text.lower()
    marker = next((m for m in ABSTENTION_MARKERS if lowered.startswith(m)), None)
    if marker is None:
        return "none"
    remainder = text
    for candidate in ABSTENTION_MARKERS:
        remainder = re.sub(re.escape(candidate), " ", remainder, flags=re.IGNORECASE)
    remainder = remainder.strip(" \t\n\r.,;:*_-")
    return "partial" if len(remainder) >= PARTIAL_ANSWER_MIN_CHARS else "complete"


def diagnose(result: RagasResult) -> tuple[str, str]:
    """Evidence-based root cause for one row, with `DIAGNOSTIC_TREE` as the fallback.

    Decision order, most specific and verifiable first:
      1. version conflict — an obsolete document was retrieved and no in-force one was;
      2. partial answer — substantive content plus an abstention, so the fix is narrower;
      3. complete abstention — split by whether the retrieved set covers the reference
         topic at all (missing context evidence) or merely lacks specific values
         (missing specific evidence), else abstention despite available evidence;
      4. `DIAGNOSTIC_TREE[worst_metric]` for rows that neither abstained nor hit a
         version trap.

    Labels are hedged where the signal is weak ("possible"), because a lexical proxy
    cannot prove a semantic cause. Polarity errors are not auto-classified — see the note
    by the vocabulary constants.
    """
    missing = missing_reference_numbers(result)
    coverage = reference_coverage(result)
    kind = abstention_kind(result.answer)

    if is_version_conflict(result):
        retrieved = sorted(retrieved_document_versions(result))
        wanted = sorted(current_versions(result.ground_truth))
        mention_note = ""
        mentions = version_mentions_beyond_headers(result)
        if mentions:
            mention_note = (f"; phiên bản {sorted(mentions)} chỉ được NHẮC ĐẾN trong thân tài liệu cũ, "
                            "không phải tài liệu đã được truy xuất")
        return (f"Version conflict — chỉ truy xuất được tài liệu bản {retrieved} (đã hết hiệu lực), "
                f"thiếu tài liệu bản {wanted} hiện hành{mention_note}",
                "Lọc theo phiên bản khi truy vấn nhắc tới phiên bản (metadata filter trên "
                "auto_metadata.version), ưu tiên bản hiện hành, và nêu rõ hiện hành so với "
                "đã bị thay thế trong prompt generation.")

    if kind == "partial":
        # Absent-verbatim is not the same as missing evidence: a value can be derivable
        # from the context (row 33's "19 ngày" is 15 + 4 and is absent verbatim), so the
        # label reports what is absent and defers the "which half lacks evidence" call to
        # the reader instead of asserting it.
        detail = (f"các giá trị reference không xuất hiện nguyên văn trong context: "
                  f"{sorted(missing)[:4]}" if missing
                  else "chưa xác định được giá trị nào thiếu bằng chứng")
        return (f"Partial answer — đã trả lời một phần câu hỏi; {detail} "
                f"(giá trị suy ra được thì không phải thiếu bằng chứng)",
                "Đọc answer/contexts để xác định phần chưa trả lời, rồi bổ sung bằng chứng cho "
                "đúng phần đó (chunking/top-k cho bảng và phụ cấp); không hạ temperature vì "
                "phần đã trả lời là đúng.")

    if kind == "complete":
        if missing and coverage < COVERAGE_FLOOR:
            return (f"Missing context evidence — model từ chối và context gần như lệch chủ đề "
                    f"(coverage {coverage:.2f} < {COVERAGE_FLOOR}); thiếu {len(missing)} dữ kiện "
                    f"của reference (ví dụ {sorted(missing)[:3]})",
                    "Retrieval là nút cổ chai: cải thiện query/chunking/top-k để lấy đúng tài liệu "
                    "chủ đề, thêm BM25 cho số và mã, và chỉ cho phép 'Không tìm thấy.' khi mọi "
                    "context đều không liên quan.")
        if missing:
            return (f"Missing specific evidence — tài liệu liên quan đã được truy xuất "
                    f"(coverage {coverage:.2f}) nhưng thiếu {len(missing)} giá trị của reference "
                    f"(ví dụ {sorted(missing)[:3]})",
                    "Tăng recall trong nội bộ tài liệu đúng: chunk nhỏ hơn/overlap lớn hơn cho "
                    "bảng và ngưỡng, thêm BM25 cho số, và kiểm tra chunk chứa ngưỡng có được index.")
        if coverage >= COVERAGE_FLOOR:
            return (f"Abstention despite available evidence — coverage {coverage:.2f}, mọi giá trị "
                    f"đặc trưng của reference đều có trong context nhưng model vẫn từ chối",
                    "Đây là lỗi generation: nới điều kiện trả lời 'Không tìm thấy.' trong system "
                    "prompt, yêu cầu dùng mọi context liên quan (không phải hallucination).")
        return ("Abstention — chưa đủ bằng chứng để kết luận nguyên nhân (possible generation "
                "hoặc retrieval failure)",
                "Đọc thủ công answer/contexts/reference của dòng này trước khi sửa; kiểm tra cả "
                "prompt grounding và chất lượng retrieval.")

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
