"""Task 3 diagnosis: the evidence-based refinements, pinned by scenario.

Every case is built from a synthetic `RagasResult`, so these tests describe BEHAVIOUR and
never depend on the committed dataset or on particular question ids. The rules they lock
down were each found by measurement on the real run:

* a total abstention is not "LLM hallucinating" (the metric-only mapping said so);
* an answer that opens with an abstention but keeps answering is a PARTIAL answer
  (the plain startswith test called it a total abstention);
* a marker buried inside a substantive answer is NOT an abstention
  (row 39's comparison table contains a "không đề cập" cell and was mislabelled);
* a retrieved document's own version comes from its m5 header, not from a version-like
  string in the body — an obsolete document can name its own replacement
  (row 44's v1.0 context says "ĐÃ THAY THẾ bởi v2.0");
* a mere mention must not suppress a version-conflict label, and context versions must
  not be split on the decimal point (a "v2.0" version label used to become "v2" + "0").
"""
from __future__ import annotations

import pytest

from src.phase_a_ragas import (DIAGNOSTIC_TREE, METRICS, RagasResult, abstention_kind, bottom_10,
                               content_tokens, context_document_version, current_versions,
                               diagnose, is_version_conflict, missing_reference_numbers,
                               obsolete_versions, reference_coverage, retrieved_document_versions,
                               version_markers, version_mentions_beyond_headers)

pytestmark = pytest.mark.student


def make(answer: str, ground_truth: str, contexts: list[str], *,
         question: str = "Câu hỏi kiểm thử?", faithfulness: float = 0.5) -> RagasResult:
    return RagasResult(1, "factual", question, answer, contexts, ground_truth,
                       faithfulness, 0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Abstention: complete vs partial vs none
# ---------------------------------------------------------------------------


def test_total_abstention_is_complete_not_hallucinating():
    result = make("Không tìm thấy.", "Chính sách cho phép 3 ngày.", ["context không liên quan"])
    assert abstention_kind(result.answer) == "complete"
    diagnosis, fix = diagnose(result)
    assert "hallucinat" not in diagnosis.lower(), "a refusal must never be called hallucination"
    assert fix.strip()


def test_partial_abstention_is_distinguished_from_total():
    answer = ("Không tìm thấy thông tin về tổng phụ cấp hàng tháng trong context. "
              "Về số ngày phép năm theo v2024: chính sách quy định 15 ngày phép cơ bản, "
              "cộng 1 ngày cho mỗi 3 năm thâm niên, nên 12 năm thâm niên được 19 ngày phép.")
    result = make(answer, "Phụ cấp 1.500.000 VNĐ và 19 ngày phép.",
                  ["[nghỉ phép | quy định | v2024] 15 ngày phép cơ bản, thâm niên cộng thêm 1 ngày mỗi 3 năm"])
    assert abstention_kind(result.answer) == "partial"
    assert diagnose(result)[0].startswith("Partial answer")


def test_marker_inside_a_substantive_answer_is_not_an_abstention():
    """Row 39's shape: a comparison table legitimately containing a "không đề cập" cell."""
    answer = ("So sánh theo context: | Tiêu chí | v1.0 | v2.0 | | Độ dài tối thiểu | 8 ký tự | 12 ký tự | "
              "| MFA | không đề cập | bắt buộc | Kết luận: v2.0 siết chặt hơn về độ dài và MFA.")
    result = make(answer, "v1.0: 8 ký tự, không MFA. v2.0: 12 ký tự, bắt buộc MFA.",
                  ["[mật khẩu | quy định | v1.0] tối thiểu 8 ký tự",
                   "[mật khẩu | quy định | v2.0] tối thiểu 12 ký tự, bắt buộc MFA"])
    assert abstention_kind(result.answer) == "none"
    assert diagnose(result) == DIAGNOSTIC_TREE[result.worst_metric]


def test_complete_abstention_with_full_evidence_points_at_generation():
    result = make("Không tìm thấy.",
                  "Cấm dùng VPN cá nhân (NordVPN, ExpressVPN); bắt buộc dùng VPN công ty.",
                  ["[VPN | quy định] Không sử dụng VPN cá nhân (NordVPN, ExpressVPN) trong mạng công ty."])
    assert reference_coverage(result) >= 0.30
    assert missing_reference_numbers(result) == set()
    assert diagnose(result)[0].startswith("Abstention despite available evidence")


def test_complete_abstention_with_off_topic_context_points_at_retrieval():
    result = make("Không tìm thấy.",
                  "NÊN: báo cáo sự cố trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline (ext. 9999). "
                  "KHÔNG được: tự xử lý malware hay xóa dấu vết; vi phạm có thể bị kỷ luật.",
                  ["[thử việc | quy định | 1.2] Thời gian thử việc tiêu chuẩn là 60 ngày.",
                   "[phân loại dữ liệu | quy định] Quy định xử lý dữ liệu Tối mật gồm mã hóa end-to-end."])
    assert reference_coverage(result) < 0.30, "off-topic retrieval must score low coverage"
    assert diagnose(result)[0].startswith("Missing context evidence")


def test_complete_abstention_with_missing_value_but_right_topic_is_specific_evidence():
    result = make("Không tìm thấy.",
                  "Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc phê duyệt.",
                  ["[mua sắm | quy trình | 2.2] Thẩm quyền phê duyệt đơn hàng dưới 5.000.000 VNĐ thuộc "
                   "Trưởng phòng; đơn hàng trên 10.000.000 VNĐ cần 3 báo giá và chờ phê duyệt theo thẩm quyền."])
    assert reference_coverage(result) >= 0.30
    assert "50000000" in missing_reference_numbers(result)
    assert diagnose(result)[0].startswith("Missing specific evidence")


# ---------------------------------------------------------------------------
# Versions: determined from the retrieved document, not from body strings
# ---------------------------------------------------------------------------


def test_version_markers_are_not_split_by_the_sentence_splitter():
    """Regression: `v2.0 HIỆN HÀNH` used to split into `v2` and `0`, emptying obsolete."""
    reference = "Theo v2.0 HIỆN HÀNH: mỗi 120 ngày. Chính sách v1.0 (đã hết hiệu lực) là 90 ngày."
    assert version_markers(reference) == {"2.0", "1.0"}
    assert obsolete_versions(reference) == {"1.0"}
    assert current_versions(reference) == {"2.0"}


def test_context_document_version_reads_the_m5_header_only():
    assert context_document_version("[mật khẩu | quy định | v1.0] nội dung") == "1.0"
    assert context_document_version("[mật khẩu | quy định] nội dung") is None
    assert context_document_version("không có header nào") is None


def test_old_document_mentioning_a_newer_version_is_still_obsolete_only():
    reference = "Theo v2.0 HIỆN HÀNH: mỗi 120 ngày. Chính sách v1.0 (đã hết hiệu lực) là 90 ngày."
    result = make("Không tìm thấy.", reference,
                  ["[mật khẩu | quy định | v1.0] Phiên bản 1.0 yêu cầu mật khẩu tối thiểu 8 ký tự. "
                   "Trạng thái: ĐÃ THAY THẾ bởi v2.0",
                   "[phân loại sự cố bảo mật | quy trình] Sự cố trung bình xử lý trong 8 giờ."])
    assert retrieved_document_versions(result) == {"1.0"}
    assert version_mentions_beyond_headers(result) == {"2.0"}
    assert is_version_conflict(result) is True
    diagnosis, fix = diagnose(result)
    assert diagnosis.startswith("Version conflict")
    assert "NHẮC ĐẾN" in diagnosis, "the body mention must be called out, not counted as retrieved"
    assert fix.strip()


def test_retrieving_both_versions_is_not_a_conflict():
    reference = ("v1.0 (đã hết hiệu lực): 8 ký tự, đổi mỗi 90 ngày. "
                 "v2.0 (hiện hành): 12 ký tự, đổi mỗi 120 ngày.")
    result = make("So sánh: v1.0 8 ký tự / 90 ngày; v2.0 12 ký tự / 120 ngày.", reference,
                  ["[mật khẩu | quy định | v1.0] tối thiểu 8 ký tự, đổi mỗi 90 ngày",
                   "[mật khẩu | quy định | v2.0] tối thiểu 12 ký tự, đổi mỗi 120 ngày"])
    assert retrieved_document_versions(result) == {"1.0", "2.0"}
    assert is_version_conflict(result) is False


def test_no_versions_in_reference_means_no_version_conflict():
    result = make("Không tìm thấy.", "Chính sách nghỉ phép là 15 ngày.",
                  ["[nghỉ phép | quy định | v1.0] 12 ngày phép"])
    assert is_version_conflict(result) is False


# ---------------------------------------------------------------------------
# Polarity: an answer containing a negation is not automatically an error
# ---------------------------------------------------------------------------


def test_answer_with_negation_is_not_flagged_as_a_polarity_error():
    result = make("Có. Tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ.",
                  "Có, theo v2.0 HIỆN HÀNH, MFA bắt buộc cho tất cả nhân viên.",
                  ["[mật khẩu | quy định | v2.0] bắt buộc MFA cho email, VPN, hệ thống nội bộ"])
    diagnosis, fix = diagnose(result)
    assert "negation" not in diagnosis.lower()
    assert "polarity" not in diagnosis.lower()
    assert (diagnosis, fix) == DIAGNOSTIC_TREE[result.worst_metric]


# ---------------------------------------------------------------------------
# Helpers and contract invariants
# ---------------------------------------------------------------------------


def test_coverage_helpers_ignore_common_words():
    assert content_tokens("nhân viên của và ngày") == set()
    assert "mfa" in content_tokens("MFA bắt buộc")


def test_coverage_is_one_when_the_reference_has_no_topical_tokens():
    result = make("x", "của và là", ["bất kỳ"])
    assert reference_coverage(result) == 1.0


def test_diagnosis_never_returns_empty_text():
    for answer in ("Không tìm thấy.", "Đã trả lời đầy đủ.", "Không tìm thấy. " + "x" * 200):
        diagnosis, fix = diagnose(make(answer, "Chính sách A là 10 ngày.", ["[a | quy định] 10 ngày"]))
        assert isinstance(diagnosis, str) and diagnosis.strip()
        assert isinstance(fix, str) and fix.strip()


def test_bottom_10_ranking_is_independent_of_the_diagnosis():
    """Only annotation text may change; rank/order/worst_metric stay metric-driven."""
    results = [RagasResult(i + 1, "factual", f"Q{i}", "A", ["C"], "GT",
                           i / 100, i / 100, i / 100, i / 100) for i in range(12)]
    rows = bottom_10(results)
    assert [row["rank"] for row in rows] == list(range(1, 11))
    assert [row["question_id"] for row in rows] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert [row["avg_score"] for row in rows] == sorted(row["avg_score"] for row in rows)
    for row, result in zip(rows, results[:10]):
        assert row["worst_metric"] == min(METRICS, key=lambda metric: getattr(result, metric))
