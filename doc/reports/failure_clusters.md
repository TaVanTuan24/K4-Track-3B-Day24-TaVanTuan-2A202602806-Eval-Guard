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

Mười dòng dưới đây lấy từ Task 3; `rank`, `id`, `distribution`, `avg_score` và
`worst_metric` được exporter giữ nguyên và đã được kiểm chứng lại là không đổi.

| Rank | Id | Distribution | avg_score | worst_metric | diagnosis | suggested_fix |
|---:|---:|---|---:|---|---|---|
| 1 | 5 | factual | 0.0000 | faithfulness | Retrieval failure + abstention: thiếu 1 dữ kiện (50000000) | Tăng recall: chunking/top-k, thêm BM25 cho số và mã, chỉ trả lời 'Không tìm thấy.' khi mọi context không liên quan |
| 2 | 40 | multi_hop | 0.0000 | faithfulness | Retrieval failure + abstention: thiếu 1 dữ kiện (9999) | Tăng recall: chunking/top-k, thêm BM25 cho số và mã, chỉ trả lời 'Không tìm thấy.' khi mọi context không liên quan |
| 3 | 44 | adversarial | 0.0000 | faithfulness | Retrieval failure + abstention: thiếu 2 dữ kiện (120, 90) | Tăng recall: chunking/top-k, thêm BM25 cho số và mã, chỉ trả lời 'Không tìm thấy.' khi mọi context không liên quan |
| 4 | 50 | adversarial | 0.2708 | faithfulness | Abstention: context có bằng chứng nhưng model vẫn từ chối | Nới điều kiện trả lời 'Không tìm thấy.', yêu cầu dùng mọi context liên quan; đây là lỗi generation, không phải hallucination |
| 5 | 30 | multi_hop | 0.3750 | faithfulness | Retrieval failure + abstention: thiếu 1 dữ kiện (200) | Tăng recall: chunking/top-k, thêm BM25 cho số và mã, chỉ trả lời 'Không tìm thấy.' khi mọi context không liên quan |
| 6 | 33 | multi_hop | 0.4286 | answer_relevancy | Retrieval failure + abstention: thiếu 4 dữ kiện (1000000, 1500000, 19) | Tăng recall: chunking/top-k, thêm BM25 cho số và mã, chỉ trả lời 'Không tìm thấy.' khi mọi context không liên quan |
| 7 | 39 | multi_hop | 0.4503 | context_precision | Too many irrelevant chunks | Add reranking or metadata filter |
| 8 | 21 | multi_hop | 0.4722 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 9 | 6 | factual | 0.5597 | context_precision | Too many irrelevant chunks | Add reranking or metadata filter |
| 10 | 22 | multi_hop | 0.5833 | answer_relevancy | Answer doesn't match question | Improve prompt template |

Chuỗi `diagnosis`/`suggested_fix` đầy đủ (kèm danh sách dữ kiện thiếu) nằm trong
`artifacts/ragas_50q.json`; bảng trên ghi đúng nội dung đó ở dạng rút gọn.

### Vì sao diagnosis không còn theo worst_metric thuần

`worst_metric` (metric thấp nhất của câu) vẫn được giữ đúng theo hợp đồng Task 3 và
là argmin của bốn metric. Nhưng ánh xạ thuần `worst_metric → nguyên nhân` gán nhãn
**“LLM hallucinating”** cho mọi câu có faithfulness thấp, kể cả khi câu trả lời chỉ
là “Không tìm thấy.”. Một câu từ chối trả lời **không chứa khẳng định nào không được
hỗ trợ**, nên nó là hiện tượng ngược lại với hallucination; gọi nó là hallucination
là chẩn đoán sai và sẽ dẫn tới hành động sửa sai (hạ temperature) trong khi lỗi thật
nằm ở retrieval hoặc ở điều kiện từ chối. Task 3 vì vậy phân loại theo bằng chứng
theo thứ tự: xung đột phiên bản (chỉ khi chứng minh được), rồi tới từ chối trả lời,
rồi mới tới bảng mặc định theo metric. Không có metric, thứ hạng hay thành viên
bottom 10 nào bị thay đổi.

### Nguyên nhân thật của từng nhóm

1. **Từ chối trả lời + thiếu dữ kiện (id 5, 40, 44, 30, 33).** Cả năm câu trả lời
   đúng là “Không tìm thấy.”, và phép kiểm tra dữ kiện cho thấy các **giá trị đặc
   trưng của reference không hề xuất hiện trong context**: id 5 thiếu `50.000.000`,
   id 40 thiếu `9999`, id 44 thiếu cả `120` lẫn `90`, id 30 thiếu `200`, id 33 thiếu
   `1.000.000`, `1.500.000` và `19`. Với id 5, contexts lấy về có `dưới 5.000.000` và
   `trên 10.000.000` nhưng **không có dòng ngưỡng 50.000.000** — bằng chứng trực tiếp
   rằng chunk chứa ngưỡng cần thiết chưa được truy xuất, nên đây là **retrieval
   failure**, không phải lỗi prompt như kết luận trước. Cách sửa đúng là tăng recall
   (chunking/top-k, thêm BM25 cho số và mã), không phải hạ temperature.
2. **Từ chối trả lời dù context đã có bằng chứng (id 50).** Câu hỏi là nhân viên
   Manager có được dùng VPN cá nhân khi WFH. Cả `ctx0` và `ctx2` đều nêu rõ
   “Không sử dụng VPN cá nhân (NordVPN, ExpressVPN) trong mạng công ty”, tức là
   **quy định cấm đã nằm trong context** nhưng model vẫn trả lời “Không tìm thấy.”.
   Đây là lỗi generation (điều kiện từ chối quá chặt), cần sửa prompt grounding.
3. **id 39 (so sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0)** có faithfulness
   1.00 và answer_relevancy 0.80 nhưng context_precision và context_recall đều 0.00:
   câu trả lời dạng bảng bám context, song context lấy về không phủ hết các ý của
   reference — đúng chữ ký của bẫy so sánh phiên bản.
4. **id 6 (thông tin lương thuộc cấp độ phân loại dữ liệu nào)** trả lời đúng
   (cấp 3 — Bí mật) với faithfulness 0.50 nhưng context_precision 0.33: chunk
   retrieval thứ hai và thứ ba không đóng góp, tức là vấn đề precision của top-3
   chứ không phải nội dung.
5. **Nhóm lệch trọng tâm (id 21, 22).** `worst_metric` là answer_relevancy:
   câu trả lời đúng nội dung nhưng không khớp trọng tâm câu hỏi nhiều ý, đúng như
   ánh xạ mặc định “Answer doesn't match question”.

**Kiểm tra polarity (negation) bằng cách đọc, không bằng từ khoá.** Task 3 **không**
tự động phân loại lỗi đảo phủ định. Một luật từ khoá kiểu “reference có `cấm` mà
answer không có từ phủ định thì là lỗi” đã được thử và **gán sai cho id 45**, một câu
trả lời đúng (faithfulness 1.0: “Có. Tất cả nhân viên bắt buộc kích hoạt MFA…” khớp
reference). Vì luật đó tạo dương tính giả trên chính bộ dữ liệu này nên nó đã bị loại
bỏ; polarity được kiểm tra bằng cách đọc dòng và được ghi nhận là **không có dòng nào
trong bottom 10 là lỗi polarity**.

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
thấp (0.6623). Trong bottom 10 có **từ chối trả lời** (id 5, 40, 44, 30, 33, 50),
**lệch trọng tâm nhiều ý** (id 21, 22) và **precision của top-3** (id 6, 39). Điểm
thấp hơn **không tự chứng minh** pipeline nhận ra bẫy; dưới đây là những gì bằng
chứng thực sự cho phép kết luận.

**Về bẫy phiên bản, kết luận phải hẹp hơn lần trước.** Chỉ 3 câu trong 50 có
reference phân biệt rõ bản hết hiệu lực với bản hiện hành: id 41, 42 và 49 (bẫy
v2023 so với v2024). Kiểm tra bằng chứng cho thấy:

- id 41 và id 49 lấy về context có **cả hai** mốc phiên bản → đây là hành vi đúng
  cho câu hỏi so sánh, không phải lấy nhầm bản cũ;
- id 42 lấy về context **không có mốc phiên bản nào**;
- do đó tiêu chí chặt “chỉ lấy được bản đã hết hiệu lực và thiếu hẳn bản hiện hành”
  **không kích hoạt ở câu nào trong lần chạy này**, và Task 3 không gán nhãn
  “Version conflict” cho dòng nào.

Riêng id 44 (thời hạn đổi mật khẩu) là **retrieval failure**, không phải xung đột
phiên bản như kết luận trước: context có chuỗi `v2.0` nhưng cả hai giá trị `120` và
`90` của reference đều **không** xuất hiện, tức là chunk chứa quy định thời hạn đổi
mật khẩu chưa được truy xuất — việc context lấy về mang tag `v1.0` chỉ là triệu
chứng đi kèm. Bằng chứng cho thấy pipeline **chưa** phân biệt được hiện hành với
hết hiệu lực, nhưng cách nó sai là *thiếu dữ kiện*, và đó là điều cần sửa.

Giới hạn của phép đo: 50 câu, một generation model duy nhất, một evaluator LLM; mỗi
câu chỉ 3 context nên context_precision/recall rất nhạy với một chunk sai. 21 trong
50 câu có ít nhất một giá trị đặc trưng của reference không xuất hiện trong context,
nên retrieval là nút cổ chai chính chứ không phải chất lượng sinh văn bản. Corpus
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
