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
| 1 | 5 | factual | 0.0000 | faithfulness | Missing specific evidence: tài liệu đúng đã được truy xuất (coverage 0.60) nhưng thiếu 1 giá trị (50000000) | Tăng recall trong nội bộ tài liệu đúng: chunk nhỏ hơn/overlap lớn hơn cho bảng và ngưỡng, thêm BM25 cho số |
| 2 | 40 | multi_hop | 0.0000 | faithfulness | Missing context evidence: context gần như lệch chủ đề (coverage 0.12) và thiếu 1 dữ kiện (9999) | Retrieval là nút cổ chai: cải thiện query/chunking/top-k để lấy đúng tài liệu chủ đề, thêm BM25 cho số và mã |
| 3 | 44 | adversarial | 0.0000 | faithfulness | Version conflict: chỉ truy xuất được tài liệu bản 1.0 (hết hiệu lực), thiếu bản 2.0 hiện hành | Lọc theo phiên bản (metadata filter trên auto_metadata.version), ưu tiên bản hiện hành |
| 4 | 50 | adversarial | 0.2708 | faithfulness | Abstention despite available evidence: coverage 0.39, mọi giá trị của reference đều có trong context | Lỗi generation: nới điều kiện trả lời 'Không tìm thấy.', yêu cầu dùng mọi context liên quan |
| 5 | 30 | multi_hop | 0.3750 | faithfulness | Missing specific evidence: tài liệu đúng đã được truy xuất (coverage 0.54) nhưng thiếu 1 giá trị (200) | Tăng recall trong nội bộ tài liệu đúng: chunk nhỏ hơn/overlap lớn hơn cho bảng, thêm BM25 cho số |
| 6 | 33 | multi_hop | 0.4286 | answer_relevancy | Partial answer: đã trả lời một phần; giá trị không xuất hiện nguyên văn: 1000000, 1500000, 19, 500000 | Đọc answer/contexts để xác định phần chưa trả lời rồi bổ sung bằng chứng cho đúng phần đó |
| 7 | 39 | multi_hop | 0.4503 | context_precision | Too many irrelevant chunks | Add reranking or metadata filter |
| 8 | 21 | multi_hop | 0.4722 | answer_relevancy | Answer doesn't match question | Improve prompt template |
| 9 | 6 | factual | 0.5597 | context_precision | Too many irrelevant chunks | Add reranking or metadata filter |
| 10 | 22 | multi_hop | 0.5833 | answer_relevancy | Answer doesn't match question | Improve prompt template |

Chuỗi `diagnosis`/`suggested_fix` đầy đủ nằm trong `artifacts/ragas_50q.json`; bảng
trên ghi đúng nội dung đó ở dạng rút gọn.

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

Task 3 phân loại theo tín hiệu có thể kiểm chứng, và ghi rõ **độ mạnh của bằng chứng**:
`coverage` (tỉ lệ từ khoá chủ đề của reference xuất hiện trong context, đã loại từ
phổ biến), các **giá trị đặc trưng** không xuất hiện nguyên văn, và **phiên bản của
chính tài liệu được truy xuất** đọc từ header m5 `[chủ đề | loại | phiên bản]`.

1. **Thiếu bằng chứng trong nội bộ tài liệu đúng (id 5, 30).** Tài liệu lấy về
   **đúng chủ đề** (coverage 0.60 và 0.54) nhưng thiếu đúng giá trị cần: id 5 thiếu
   `50.000.000` — contexts có `dưới 5.000.000` và `trên 10.000.000` nhưng **không có
   dòng ngưỡng 50.000.000**; id 30 thiếu `200` (hạn mức PVI). Vì tài liệu đã đúng
   chủ đề, đây là lỗi **recall trong nội bộ tài liệu** (chunking/overlap/BM25 cho số),
   không phải lỗi chọn tài liệu và cũng không phải lỗi prompt.
2. **Context lệch chủ đề (id 40).** `coverage` chỉ **0.12** — thấp hơn hẳn mọi dòng
   khác (các dòng có bằng chứng đều ≥ 0.39). Câu hỏi hỏi về **quy trình báo cáo sự cố
   bảo mật** (báo cáo trong 1 giờ, kênh nào, không được tự xử lý, hậu quả), nhưng cả ba
   context là `thử việc | quy định | 1.2`, `thử việc | quy định` và
   `phân loại dữ liệu | quy định` — **không có tài liệu nào về báo cáo sự cố**. Thiếu
   `9999` chỉ là hệ quả; bằng chứng chính là chủ đề lệch hoàn toàn, nên đây là lỗi
   **retrieval ở tầng chọn tài liệu**, và cách sửa là query/chunking/top-k chứ không
   phải hạ temperature.
3. **Xung đột phiên bản (id 44).** Đây là dòng duy nhất trong bottom 10 bị phân loại
   version conflict, và **kết luận của lần review trước đã bị đảo lại có căn cứ**:
   tài liệu được truy xuất mang header `mật khẩu | quy định | v1.0`, tức là **bản 1.0
   đã hết hiệu lực**; chuỗi `v2.0` xuất hiện trong context **chỉ vì bản 1.0 tự ghi
   “Trạng thái: ĐÃ THAY THẾ bởi v2.0”**, chứ không phải tài liệu 2.0 đã được lấy về.
   Vì vậy việc dùng regex tìm chuỗi `v2.0` là bằng chứng sai, còn header của tài liệu
   là bằng chứng đúng; đối chiếu `obsolete = {1.0}`, `current = {2.0}`,
   `retrieved = {1.0}` cho ra obsolete-only retrieval. Cách sửa: lọc theo phiên bản khi
   truy vấn nhắc tới phiên bản. Lưu ý phân biệt: id 41 và id 49 cũng có reference nói
   về v2023/v2024 nhưng **lấy về cả hai bản** (đúng cho câu hỏi so sánh) nên **không**
   bị gọi là conflict.
4. **Từ chối trả lời dù context đã có bằng chứng (id 50).** `coverage` 0.39 và **mọi
   giá trị đặc trưng của reference đều có trong context**: cả `ctx0` và `ctx2` nêu rõ
   “Không sử dụng VPN cá nhân (NordVPN, ExpressVPN) trong mạng công ty”. Model vẫn trả
   lời “Không tìm thấy.”, nên đây là lỗi **generation** (điều kiện từ chối quá chặt).
5. **Trả lời một phần (id 33).** Đây **không phải** total abstention như kết luận
   trước: câu trả lời mở đầu bằng “Không tìm thấy thông tin về tổng phụ cấp…” nhưng
   **tiếp tục trả lời đúng phần phép năm** (“15 + 4 = 19 ngày phép”, dựa trên context
   v2024). Phần thiếu thật là **phụ cấp** (`1.000.000` và `500.000` không có trong
   context); riêng `19` không xuất hiện nguyên văn vì **phải suy ra** từ 15 + 4, nên nó
   không phải bằng chứng thiếu. Vì vậy nhãn đúng là `Partial answer`, và cách sửa là bổ
   sung bằng chứng cho đúng phần chưa trả lời.
6. **id 39 (so sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0)** có faithfulness
   1.00 và answer_relevancy 0.80 nhưng context_precision và context_recall đều 0.00:
   câu trả lời dạng bảng bám context, song context lấy về không phủ hết các ý của
   reference. Bảng của câu trả lời có ô “không đề cập”, nhưng đó **không** phải từ chối
   trả lời — Task 3 chỉ coi là abstention khi câu trả lời **mở đầu** bằng câu từ chối,
   nên id 39 giữ nguyên chẩn đoán theo metric (`Too many irrelevant chunks`).
7. **id 6** trả lời đúng (cấp 3 — Bí mật) với faithfulness 0.50 nhưng
   context_precision 0.33: chunk thứ hai và thứ ba không đóng góp, tức là vấn đề
   precision của top-3 chứ không phải nội dung.
8. **Nhóm lệch trọng tâm (id 21, 22).** `worst_metric` là answer_relevancy: câu trả lời
   đúng nội dung nhưng không khớp trọng tâm câu hỏi nhiều ý, đúng như ánh xạ mặc định
   “Answer doesn't match question”.

**Giới hạn của phân loại tự động.** `coverage` là proxy từ khoá, không phải ngữ nghĩa;
ngưỡng `0.30` được chọn từ khoảng trống đo được (0.12 so với 0.39–0.85) và được ghi rõ
trong code, không phải suy diễn chắc chắn. Vì vậy nơi tín hiệu yếu, Task 3 dùng chữ
“possible/chưa đủ bằng chứng để kết luận” thay vì khẳng định, và phần đọc định tính
dưới đây là nguồn kết luận chính.

**Kiểm tra polarity (negation) bằng cách đọc, không bằng từ khoá.** Task 3 **không**
tự động phân loại lỗi đảo phủ định. Một luật từ khoá kiểu “reference có `cấm` mà
answer không có từ phủ định thì là lỗi” đã được thử và **gán sai cho id 45**, một câu
trả lời đúng (faithfulness 1.0: “Có. Tất cả nhân viên bắt buộc kích hoạt MFA…” khớp
reference). Vì luật đó tạo dương tính giả trên chính bộ dữ liệu này nên nó đã bị loại
bỏ; polarity được kiểm tra bằng cách đọc dòng và được ghi nhận là **không có dòng nào
trong bottom 10 là lỗi polarity**. Lưu ý id 45 còn có đặc điểm “answer chứa phủ định
nhưng không sai” (khẳng định bắt buộc MFA), nên một luật polarity dựa trên từ khoá sẽ
luôn bất khả thi ở đây.

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
thấp (0.6623). Trong bottom 10 có **thiếu bằng chứng trong tài liệu đúng** (id 5, 30),
**context lệch chủ đề** (id 40), **xung đột phiên bản** (id 44), **từ chối dù có bằng
chứng** (id 50), **trả lời một phần** (id 33), **precision của top-3** (id 6, 39) và
**lệch trọng tâm nhiều ý** (id 21, 22). Điểm thấp hơn **không tự chứng minh** pipeline
nhận ra bẫy; dưới đây là những gì bằng chứng thực sự cho phép kết luận.

**Bẫy phiên bản: đúng một dòng bị gọi là conflict, và đó là id 44.** Cách xác định
phiên bản dùng **header của chính tài liệu được truy xuất**, không dùng việc tìm thấy
chuỗi phiên bản trong thân văn bản, vì một tài liệu cũ có thể tự nhắc tới bản mới:

- id 44: tài liệu lấy về có header `mật khẩu | quy định | v1.0`; chuỗi `v2.0` chỉ nằm
  trong câu “Trạng thái: ĐÃ THAY THẾ bởi v2.0” **của chính bản 1.0**. Đối chiếu
  `obsolete = {1.0}`, `current = {2.0}`, `retrieved = {1.0}` → **obsolete-only
  retrieval**, nên đây **là** xung đột phiên bản. Kết luận “retrieval failure” ở lần
  review trước dựa trên chính chuỗi `v2.0` đó và đã được sửa lại.
- id 41 và id 49: reference cũng nói về v2023/v2024 nhưng **lấy về cả hai bản**
  (`retrieved = {2023, 2024}`) → hành vi đúng cho câu hỏi so sánh, **không** gọi là
  conflict.
- id 42: context **không có mốc phiên bản nào** trong header → không đủ bằng chứng để
  kết luận conflict, nên không gán nhãn đó.

Bằng chứng cho thấy pipeline **chưa** phân biệt được hiện hành với hết hiệu lực, và
cách nó sai ở id 44 là **lấy nhầm bản cũ**, nên cách sửa đúng là lọc theo phiên bản.

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
