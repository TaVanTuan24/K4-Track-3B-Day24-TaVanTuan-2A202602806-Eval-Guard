# Phân tích lỗi Phase A

Nhóm: K4-Track-3B-TaVanTuan (bài cá nhân).

Người phụ trách: Tạ Văn Tuấn — 2A202602806.

Nguồn: `artifacts/ragas_50q.json`; run timestamp: 2026-10-12T12:56:28Z (UTC).

Điều kiện chạy thật: 50/50 answers có `generation_mode = llm` và
`generation_model = deepseek-ai/DeepSeek-V4.1-Flash` (không có extractive_fallback).
Bốn metric do đường RAGAS library thật sinh ra (0.1.22, `ChatOpenAI` +
embedding local bge-m3), không có lượt gọi nào được sinh thủ công.
Models generation/evaluator: generation = `deepseek-ai/DeepSeek-V4.1-Flash`;
evaluator = `BAAI/bge-m3`, `deepseek-ai/DeepSeek-V4.1-Flash`.

## Điểm theo distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) |
|---|---|---|---|
| faithfulness | 0.9250 | 0.6295 | 0.7667 |
| answer_relevancy | 0.8237 | 0.6111 | 0.5944 |
| context_precision | 0.8417 | 0.7833 | 0.7667 |
| context_recall | 0.7833 | 0.6250 | 0.3833 |
| avg_score | 0.8434 | 0.6623 | 0.6278 |

## Bottom 10 và nguyên nhân

Mười dòng dưới đây lấy nguyên từ Task 3 và được exporter giữ nguyên, không tính lại.

| Rank | Id | Distribution | avg_score | worst_metric | diagnosis | suggested_fix |
|---:|---:|---|---:|---|---|---|
| 1 | 5 | factual | 0.0000 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 2 | 40 | multi_hop | 0.0000 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 3 | 44 | adversarial | 0.0000 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 4 | 50 | adversarial | 0.2708 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 5 | 30 | multi_hop | 0.3750 | faithfulness | LLM hallucinating | Tighten system prompt, lower temperature |
| 6 | 33 | multi_hop | 0.4286 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 7 | 39 | multi_hop | 0.4503 | context_precision | Too many irrelevant chunks | Add reranking or metadata filter |
| 8 | 21 | multi_hop | 0.4722 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 9 | 6 | factual | 0.5597 | context_precision | Too many irrelevant chunks | Add reranking or metadata filter |
| 10 | 22 | multi_hop | 0.5833 | answer_relevancy | Answer doesn't match question | Improve prompt template |

Đối chiếu answer/contexts/reference của các câu đại diện cho thấy **hai nguyên nhân
khác nhau**, không phải một nhóm lỗi duy nhất:

1. **Câu trả lời là “Không tìm thấy.” (id 5, 40, 44, 50, 30) — lỗi generation.**
   Metric về 0 vì câu trả lời không có khẳng định nào để đối chiếu. Với id 5
   (phê duyệt mua thiết bị 55 triệu), contexts đều thuộc tài liệu `mua_sam`
   (tag `mua sắm | quy trình | 2.2`) tức là bằng chứng **có** trong context nhưng
   model vẫn từ chối trả lời. Nguyên nhân trực tiếp là system prompt grounding quá
   chặt: “Nếu context không chứa thông tin, trả lời đúng câu: Không tìm thấy.”, kết
   hợp ngưỡng phê duyệt nằm rải ở nhiều đoạn. Đây là lỗi prompt/temperature, không
   phải lỗi retrieval.
2. **Retrieval lấy sai phiên bản (id 44, 40) — lỗi retrieval.**
   Với id 44 (thời hạn đổi mật khẩu), contexts có tag `mật khẩu | quy định` và
   `mật khẩu | quy định | v1.0` nhưng **không có chunk v2.0 hiện hành**, trong khi
   ground truth yêu cầu v2.0 (120 ngày) và nói rõ v1.0 đã hết hiệu lực. Với id 40,
   contexts là `thử việc` và `phân loại dữ liệu` trong khi ground truth nằm ở quy
   trình báo cáo sự cố bảo mật (báo cáo trong 1 giờ).
3. **id 39 (so sánh mật khẩu v1.0 và v2.0)** có faithfulness 1.00 và
   answer_relevancy 0.80 nhưng context_precision và context_recall đều 0.00: câu
   trả lời dạng bảng bám context, song context lấy về không phủ hết các ý của
   reference — đúng chữ ký của bẫy so sánh phiên bản.
4. **id 6 (thông tin lương thuộc cấp độ phân loại dữ liệu nào)** trả lời đúng
   (cấp 3 — Bí mật) với faithfulness 0.50 nhưng context_precision 0.33: chunk
   retrieval thứ hai và thứ ba không đóng góp, tức là vấn đề precision của top-3
   chứ không phải nội dung.

## Matrix và nhóm yếu nhất

| worst_metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1 | 9 | 2 |
| answer_relevancy | 10 | 5 | 1 |
| context_precision | 4 | 4 | 2 |
| context_recall | 5 | 2 | 5 |

Nhóm yếu theo mean thấp nhất: adversarial; mean: 0.627762.
Worst metric xuất hiện nhiều nhất: answer_relevancy; số câu: 16.

Lưu ý cách đọc: tổng theo cột của matrix chỉ là kích thước nhóm (20/20/10) nên
không dùng để chọn nhóm yếu. Nhóm yếu được chọn bằng **avg_score trung bình thấp
nhất** (`distribution_avg_scores`: factual 0.843430, multi_hop 0.662252,
adversarial 0.627762), và đúng là adversarial — nhóm ít câu nhất nhưng chất lượng
thấp nhất. Trong nhóm đó, `context_recall` chỉ 0.3833 so với 0.7833 của factual,
tức là retrieval **thiếu** bằng chứng hiện hành chứ không phải thừa. Trên toàn bộ 50
câu, metric bị điểm thấp nhất nhiều nhất lại là `answer_relevancy` (16 câu, tập
trung ở factual 10 và multi_hop 5): câu trả lời đúng nội dung nhưng diễn đạt lệch
trọng tâm câu hỏi, đặc biệt ở các câu multi_hop hỏi nhiều ý cùng lúc.

Đề xuất sửa cụ thể, kiểm tra được bằng cách chạy lại đúng lệnh cũ và so JSON:
ưu tiên lọc theo phiên bản khi truy vấn có nhắc tới phiên bản (metadata filter trên
`auto_metadata.version` do m5 sinh), thêm một bước “hiện hành/đã hết hiệu lực” vào
prompt generation, và nới điều kiện trả lời “Không tìm thấy.” thành chỉ khi cả ba
context đều không liên quan. Sau khi sửa, chạy lại
`python scripts/run_experiments.py --phase a --live`; cache theo từng câu trong
`artifacts/cache/ragas.json` chỉ giữ những dòng có cùng câu hỏi/context nên các câu
không đổi không bị trả phí lại, còn câu đổi context sẽ được đo lại.

## Bẫy adversarial và giới hạn

So với factual (0.8434), adversarial thấp hơn rõ rệt (0.6278) và multi_hop cũng
thấp (0.6623). Trong bottom 10 có ba dạng bẫy: **xung đột phiên bản** (id 44 và 39 —
mật khẩu v1.0 đã hết hiệu lực so với v2.0), **phủ định** (id 50 — VPN cá nhân
NordVPN bị cấm, id 40 — vi phạm bảo mật phải báo cáo) và **nhiều ý cùng lúc**
(id 33, 21, 30, 22). Điểm thấp hơn **không tự chứng minh** pipeline đã nhận ra bẫy:
với id 44 nguyên nhân là retrieval lấy nhầm bản cũ, còn id 39 lại lấy đủ cả hai bản
mà vẫn rớt context_recall, nên bằng chứng chỉ ra rằng pipeline **chưa** phân biệt
được hiện hành với hết hiệu lực. Đây là kết luận từ việc đọc nội dung, không phải
suy từ con số trung bình.

Giới hạn của phép đo: 50 câu, một generation model duy nhất, một evaluator LLM; mỗi
câu chỉ 3 context nên context_precision/recall rất nhạy với một chunk sai. Corpus
có 28 file nhưng **2 file PDF chỉ có ảnh, không trích được chữ** (`BCTC.pdf` và
`Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf`), nên phần nội dung đó
không thể xuất hiện trong context — đây là giới hạn dữ liệu thật, không phải lỗi
code, và nó có thể góp phần vào các câu về phân loại dữ liệu. Reranker
cross-encoder (flashrank `ms-marco-MultiBERT-L-12`) cho điểm bão hoà trên corpus
tiếng Việt này — 20 candidate nằm trong khoảng 0.996963 đến 0.999552 và thứ tự nó
tạo ra đặt chunk `thuong_tet` lên trên chunk đúng — nên `src/m3_rerank.py` phát
hiện bão hoà và giữ nguyên thứ tự hybrid. Một chi tiết cần ghi rõ để không hiểu sai
bằng chứng: `.env` của môi trường này để `GEMINI_API_KEY` trống và trỏ
`OPENAI_BASE_URL` sang endpoint OpenAI-compatible khác, nên các số trên là của
provider đó; toàn bộ đường dây Gemini đã được implement và kiểm chứng cấu hình,
nhưng **không có lượt gọi Gemini nào được thực hiện**, và báo cáo không được phép
gán các số này cho Gemini.
