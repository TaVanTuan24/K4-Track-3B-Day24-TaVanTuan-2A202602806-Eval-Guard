# Blueprint CI/CD cho RAG và guardrails

Nhóm: K4-Track-3B-TaVanTuan (bài cá nhân).

Người phụ trách: Tạ Văn Tuấn — 2A202602806.

Đây là **đề xuất CI/CD**, không phải workflow đã được triển khai hoặc đã chạy.
Nguồn kết quả: `artifacts/ragas_50q.json`, `artifacts/judge_results.json`,
`artifacts/guard_results.json`.

Run timestamp: Phase A 2026-10-12T12:56:28Z; Phase B 2026-10-12T12:29:02Z;
Phase C 2026-10-12T12:42:45Z (UTC). Guard models: `deepseek-ai/DeepSeek-V4.1-Flash`.
Models theo từng vai trò: generation = `deepseek-ai/DeepSeek-V4.1-Flash` (từ
`answers_50q.json`); evaluator = `BAAI/bge-m3` + `deepseek-ai/DeepSeek-V4.1-Flash`
(từ m4); judge = `deepseek-ai/DeepSeek-V4.1-Flash` (từ `judge_results.json`);
guard = `deepseek-ai/DeepSeek-V4.1-Flash` (từ `config.models` của NeMo instance
dùng ở từng stage, không suy từ model judge).

## Kiến trúc và failure action

Luồng đề xuất: Input → Presidio → NeMo input → RAG (BM25 + dense + rerank) →
NeMo output → response. Request thật dừng sớm khi bị chặn; **khác** với phép đo
Task 12 luôn chạy đủ hai tầng input để đo độc lập từng tầng.

- **Presidio (PII).** Chỉ chặn khi entity nằm trong danh sách PII của bài
  (`VN_CCCD`, `VN_PHONE`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`,
  `IBAN_CODE`, `US_SSN`). Presidio còn trả về `DATE_TIME`, `NRP`, `LOCATION`,
  `PERSON`, `URL` — nếu coi tất cả là PII thì **8/8 câu sạch bị chặn** (đo được
  false_positive_rate 0.875) vì “chính sách v2024” khớp `DATE_TIME` và “ngày mai”
  khớp `DATE`. Failure action: redact bằng `<TYPE>`, ghi lại loại entity và vị trí.
- **NeMo input rail.** Quyết định lấy từ `result.log.activated_rails` (không suy từ
  từ khoá từ chối, vì câu sạch có thể chứa “xin lỗi, tôi không thể đi làm”). Khi
  chặn, bot trả câu từ chối tiếng Việt trong `rails.co`.
- **RAG.** BM25 + dense bge-m3 hợp nhất bằng Reciprocal Rank Fusion; cross-encoder
  chỉ được dùng khi nó thật sự phân biệt được (xem mục monitoring).
- **NeMo output rail.** Kiểm tra câu trả lời trong ngữ cảnh hội thoại. Output sạch
  giữ nguyên, output nhạy cảm bị thay bằng câu từ chối.
- **Secrets.** Key chỉ nằm trong `.env` (đã git-ignore) và được client dùng chung
  đọc lúc chạy; không hardcode, không ghi vào log hay report. `.env.example` chỉ
  chứa tên biến.
- **Fallback.** Generation lỗi hoặc thiếu context thì ghi
  `generation_mode = extractive_fallback` với `generation_model = null`; retrieval
  không tới được Qdrant thì chuyển sang dense in-memory và log cảnh báo; reranker
  lỗi thì giữ thứ tự hybrid. Không nhánh nào tự tạo số liệu thay cho lượt gọi thật.

## Phạm vi và số đo latency

Bắt buộc Task 12: **input_layers_sequential**, Presidio + NeMo input, không
short-circuit. Engines tạo trước khi đo; hai warm-up thật bị loại khỏi mẫu;
percentile nội suy tuyến tính từ raw samples; tổng tính từ tổng từng sample.

Số mẫu timed: 20; warm-up: 2; phương pháp: linear.

| Layer / scope | P50 ms | P95 ms | P99 ms | Tình trạng |
|---|---:|---:|---:|---|
| Presidio | 19.26 | 23.48 | 23.57 | Đo thật, đạt |
| NeMo input | 773.33 | 1115.69 | 1675.56 | Đo thật, vượt budget |
| Input guard sequential | 789.27 | 1134.00 | 1696.46 | Đo thật, vượt budget |
| RAG | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| NeMo output | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| Full guard / short-circuit request | chưa đo | chưa đo | chưa đo | Không suy từ input_guard_ms |

Input-layer budget target: 500ms; kết quả thực tế/budget_ok: P95 1134.00 ms,
False.

Bottleneck rõ ràng: Presidio chỉ 23.48 ms ở P95 vì chạy local, còn NeMo input
1115.69 ms ở P95 vì mỗi quyết định rail là **một lượt gọi LLM qua mạng**. Đây là
số đo thật và được giữ nguyên, không chỉnh cho đạt budget. Kế hoạch lấy số cho các
phần còn thiếu: đo RAG và NeMo output với cùng cách (tạo engine trước, 2 warm-up,
đo `perf_counter` quanh từng tầng, lưu raw samples) rồi thêm một phép đo
end-to-end có short-circuit với timestamp theo từng chặng; cả ba vẫn đang là
“chưa đo”.

## CI gates đề xuất

- RAGAS: faithfulness ≥0.75 và avg_score ≥0.65 trên 50 câu.
- Guard: attacks ≥18/20, clean 8/8 allowed, 4/4 output case đúng.
- Full guard P95 <500ms chỉ áp dụng **sau khi** thực sự đo full guard.

Mốc bắt buộc của lab vẫn là attacks ≥15/20; bonus ≥18/20 kèm clean controls đạt.
Các ngưỡng RAGAS ở trên là **đề xuất của nhóm**, không phải yêu cầu bắt buộc của đề bài.

### Đối chiếu ngưỡng đề xuất với số đo thật

Hai cách áp dụng ngưỡng cho kết quả khác nhau, nên phải nói rõ đang dùng cách nào.
Mọi giá trị dưới đây tính trực tiếp từ `artifacts/ragas_50q.json` (50 dòng
`per_question`), không lấy từ `per_distribution` đã làm tròn sẵn.

Cách A — **global gates** (trung bình trên toàn bộ 50 câu):

| Chỉ số toàn tập | Giá trị | Ngưỡng đề xuất | Kết quả |
|---|---:|---:|---|
| faithfulness | 0.775145 | ≥0.75 | Đạt |
| answer_relevancy | 0.692823 | không đặt | — |
| context_precision | 0.803333 | không đặt | — |
| context_recall | 0.640000 | không đặt | — |
| avg_score | 0.727825 | ≥0.65 | Đạt |

Cách B — **per-distribution gates** (áp riêng cho từng nhóm):

| Distribution | faithfulness | ≥0.75 | avg_score | ≥0.65 |
|---|---:|---|---:|---|
| factual (20) | 0.925000 | Đạt | 0.843430 | Đạt |
| multi_hop (20) | 0.629528 | **Không đạt** | 0.662252 | Đạt |
| adversarial (10) | 0.766667 | Đạt | 0.627762 | **Không đạt** |

Kết luận đúng: theo **cách A** pipeline đạt cả hai ngưỡng toàn tập (faithfulness
0.775145 và avg_score 0.727825). Theo **cách B** thì mỗi nhóm hỏng đúng một tiêu chí
khác nhau — multi_hop chỉ hỏng faithfulness, adversarial chỉ hỏng avg_score, còn
factual đạt cả hai. Vì vậy câu “cả multi_hop và adversarial đều có avg_score dưới
0.65” là sai và đã được sửa: multi_hop có avg_score 0.662252, **trên** ngưỡng 0.65.
Cách B khắt khe hơn nhiều vì nhóm chỉ có 10 câu thì một câu 0.0 là đủ kéo trung bình
xuống: adversarial có 3 câu avg_score 0.0 trong 10 câu.

Đề xuất dùng: **cách A làm gate chặn CI** (ổn định, đạt), và **cách B làm cảnh báo
theo nhóm** (không chặn) vì nó phát hiện đúng hai điểm yếu mà cách A che mất. Guard
thì xanh tuyệt đối theo cả hai cách: attacks 20/20, clean 8/8,
false_positive_rate 0.0, output 4/4.

Workflow chưa được triển khai. Nếu triển khai, job chỉ gọi runner đã có và để
runner tự trả nonzero:
`python scripts/run_experiments.py --phase a --live` rồi
`python scripts/run_experiments.py --phase b --live` rồi
`python scripts/run_experiments.py --phase c --live --latency-samples 20`, sau đó
`python scripts/check_lab.py` (đọc artifacts, không gọi API). Các biến kiểu `MIN_*`
chỉ là **đề xuất**: `scripts/check_lab.py` hiện kiểm tra schema, nguồn, phép tính và
ngưỡng chất lượng do `src/report_contracts.py` đặt (attacks ≥15/20 khi
`require_quality`), chứ chưa kiểm tra ngưỡng RAGAS, nên không được mô tả là đã có
quality gate cho RAGAS.

## Monitoring và cải tiến

Metric và ngưỡng cảnh báo đề xuất: (1) RAGAS 4 metric theo distribution, cảnh báo
khi avg_score một nhóm giảm quá 0.05 so với lần chạy trước; (2) attacks/clean của
bộ guard, cảnh báo khi bất kỳ câu sạch nào bị chặn vì đó là tín hiệu false positive
nguy hiểm hơn một attack lọt; (3) P95 input guard theo tầng để tách chi phí local
(Presidio) khỏi chi phí gọi mạng (NeMo); (4) κ của judge khi có thêm nhãn người;
(5) tỷ lệ câu trả lời “Không tìm thấy.” như chỉ báo retrieval/precision.

Cách lấy samples: dùng lại runner theo phase, đọc raw samples trong
`guard_results.json` và `per_question` trong `ragas_50q.json`; mỗi lần chạy là một
run timestamp riêng nên so sánh được theo thời gian. False positive xử lý bằng cách
mở rộng allowlist PII và đọc lại prompt self-check, không bằng cách hạ ngưỡng chặn
attack. Privacy: log không chứa nội dung PII gốc, chỉ loại entity và vị trí; key
không vào log. Drift: theo dõi phân bố `distribution` và phiên bản chính sách xuất
hiện trong context, vì bẫy phiên bản là nguyên nhân chính của nhóm adversarial.
Người xử lý: người tích hợp pipeline (Phase A) và người phụ trách guard (Phase C).

RAGAS mean/worst distribution: worst = adversarial 0.6278; factual tốt nhất 0.8434.

Cohen’s κ: 1.0.

Attacks passed: 20/20; clean false_positive_rate: 0.0.

Năm cải tiến có thể kiểm tra, gắn với bằng chứng của bài:

1. **Lọc theo phiên bản khi truy vấn nhắc tới phiên bản**, dùng
   `auto_metadata.version` do m5 sinh. Bằng chứng: id 44 lấy về chunk
   `mật khẩu | quy định | v1.0` mà thiếu bản v2.0 hiện hành nên context_recall của
   adversarial chỉ 0.3833.
2. **Bổ sung điều kiện “hiện hành so với hết hiệu lực” vào prompt generation.**
   Bằng chứng: 5 câu trả lời “Không tìm thấy.” có avg_score 0.0 dù context chứa
   bằng chứng (id 5 với tag `mua sắm | quy trình | 2.2`).
3. **Đo lại reranker trên tập tiếng Việt có nhãn** trước khi cho nó quyền đổi thứ
   tự. Bằng chứng: 20 candidate của `ms-marco-MultiBERT-L-12` nằm trong khoảng
   0.996963 đến 0.999552 và đặt chunk `thuong_tet` lên trên chunk đúng.
4. **Tách ngân sách latency cho tầng gọi mạng**: cache kết quả self-check input cho
   câu hỏi lặp lại, hoặc dùng model nhỏ/nội bộ cho rail. Bằng chứng: NeMo input P95
   1115.69 ms so với Presidio 23.48 ms.
5. **OCR hai PDF chỉ có ảnh** (`BCTC.pdf`, `Nghi_dinh_so_13-2023...`) rồi index lại.
   Bằng chứng: `load_documents()` báo 2 file không trích được chữ, nên nội dung đó
   chắc chắn không thể xuất hiện trong context.

Giới hạn của bộ dữ liệu đánh giá: 50 câu (20/20/10), 5 cặp A/B, 10 nhãn người,
20 attacks, 8 clean, 4 output case. Đây là bộ nhỏ để học quy trình; mọi ngưỡng
đề xuất ở trên phải được xác nhận lại trên bộ lớn hơn trước khi dùng làm cổng chặn.
