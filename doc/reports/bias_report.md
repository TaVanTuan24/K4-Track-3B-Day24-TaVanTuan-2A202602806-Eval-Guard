# Đánh giá LLM Judge Phase B

Nhóm: K4-Track-3B-TaVanTuan (bài cá nhân).

Người phụ trách: Tạ Văn Tuấn — 2A202602806.

Nguồn: `artifacts/judge_results.json`; run timestamp: 2026-10-12T12:29:02Z (UTC).

Models generation/judge: generation = `deepseek-ai/DeepSeek-V4.1-Flash`;
judge = `deepseek-ai/DeepSeek-V4.1-Flash` (cả pairwise và binary dùng cùng model,
ghi theo `metadata.models`).

## Năm cặp A/B và hai lượt

Nguồn cố định: A từ `artifacts/answers_50q.json:answer`; B từ
`fixtures/human_labels_10q.json:model_answer`; reference từ
`fixtures/test_set_50q.json:ground_truth`. Ids: 1, 5, 12, 21, 41. Không đổi nguồn
sau khi xem kết quả judge; hai lượt của một cặp dùng cùng question và reference.

| Id | Lượt 1 | Lượt 2 (đã quy về A/B gốc) | final_winner | position_consistent | scores lượt 1 (A/B) | scores lượt 2 (A/B) |
|---:|---|---|---|---|---|---|
| 1 | A | A | A | True | 1.00 / 0.80 | 1.00 / 0.70 |
| 5 | tie | A | tie | False | 0.30 / 0.30 | 0.20 / 0.00 |
| 12 | A | A | A | True | 1.00 / 0.70 | 1.00 / 0.70 |
| 21 | B | B | B | True | 0.50 / 1.00 | 0.50 / 1.00 |
| 41 | A | A | A | True | 0.80 / 0.20 | 0.75 / 0.10 |

Position bias count/rate: 1 / 0.2.
Verbosity win rate: 0.75; decisive cases: 4.

Diễn giải dựa trên nội dung, không chỉ con số:

- Cặp id 5 là cặp duy nhất không nhất quán. Lượt 1 cho cả hai tương đương (tie) vì
  “cả hai đều không chính xác”, lượt 2 sau khi hoán đổi lại chọn câu không cung cấp
  thông tin sai lệch là ít gây hại hơn. Đây là dấu hiệu judge nhạy với thứ tự khi
  hai candidate cùng kém, và đúng là trường hợp swap-and-average sinh ra `tie`.
- Verbosity 0.75 nghĩa là 3 trong 4 case decisive có câu dài hơn thắng, và cả 3 đều
  là A. Đọc lý do judge thì phần thắng này đến từ **nội dung**: id 1 nêu thêm chi
  tiết “không trừ vào phép năm”, id 12 nêu điều kiện “nhân viên chính thức từ 6
  tháng trở lên”, id 41 nêu đúng số ngày hiện hành v2024 và nói rõ v2023 đã bị thay
  thế. Vì vậy tỷ lệ này là **dấu hiệu cần đọc nội dung**, không đủ để kết luận judge
  thiên vị độ dài.
- Cặp id 41 đáng chú ý: judge nêu rõ A sai ở chi tiết thâm niên nhưng vẫn chọn A vì
  A bám đúng phiên bản hiện hành — đúng kiểu phán đoán mà bài yêu cầu.

## Calibration đúng model_answer

Helper đánh giá đúng `question`/`model_answer` đang có trong
`fixtures/human_labels_10q.json`, không lấy answer mới của pipeline. Prompt chỉ nhận
`question`, `answer`, `reference`; `human_label`, `human_note` và phân bố nhãn human
không được gửi vào prompt.

| question_id | Nhãn judge | Nhãn human | Agree | Lý do judge (rút gọn) |
|---:|---:|---:|---|---|
| 1 | 1 | 1 | yes | Đúng 3 ngày làm việc có lương khi kết hôn |
| 5 | 0 | 0 | yes | 55 triệu vượt 50 triệu nên phải CEO, không phải Giám đốc phòng ban |
| 12 | 1 | 1 | yes | Đúng mức thưởng Tết tối thiểu 1 tháng lương |
| 21 | 1 | 1 | yes | Đúng v2024: 18 ngày phép và lương Senior 20-35 triệu |
| 23 | 1 | 1 | yes | Chưa đủ 1 năm cam kết nên hoàn trả 25 triệu |
| 29 | 0 | 0 | yes | Thiếu Kế toán trưởng và số tiền phạt cụ thể |
| 33 | 1 | 1 | yes | Tổng phụ cấp 1.500.000 VNĐ/tháng và 19 ngày phép |
| 41 | 0 | 0 | yes | 12 ngày là chính sách cũ v2023 đã bị thay thế |
| 46 | 1 | 1 | yes | Thử việc không được nghỉ phép năm |
| 50 | 0 | 0 | yes | VPN v1.3 cấm VPN cá nhân nên “Được” là sai |

Cohen’s κ: 1.0.

Observed agreement = 10/10 = 1.000. Nhãn human có 6 nhãn 1 và 4 nhãn 0; judge cho
đúng cùng phân bố nên chance agreement
p_e = (6/10 · 6/10) + (4/10 · 4/10) = 0.36 + 0.16 = 0.520, và
κ = (1.000 − 0.520) / (1 − 0.520) = 1.000. Theo thang Landis-Koch đây là
“almost perfect”, và đáng chú ý là judge bắt đúng cả ba loại lỗi khó: xung đột phiên
bản (id 41), phủ định (id 50) và ngưỡng số (id 5, 29).

## Giới hạn và cách dùng judge

Bốn giới hạn phải nói rõ để không thổi phồng κ = 1.0:

1. **Mẫu 10 nhãn là rất nhỏ.** Một nhãn đổi chỗ có thể làm κ rơi xuống khoảng 0.8,
   nên không thể coi 1.0 là độ tin cậy ổn định của judge.
2. **Judge được cấp `reference` (ground truth) và nhãn human cũng được tạo theo
   reference**, nên κ ở đây đo “model có tái lập được phán đoán dựa trên reference
   hay không”, chứ không đo khả năng chấm điểm tốt/xấu khi không có đáp án.
3. **Cả hai lượt judge dùng cùng một model** và cùng prompt; đổi model judge có thể
   đổi kết quả mà không có bộ nhãn thứ hai để đối chiếu.
4. Pairwise và nhãn 0/1 trả lời hai câu hỏi khác nhau: winner A/B/tie là lựa chọn
   giữa hai candidate, còn nhãn 0/1 là đánh giá một answer so với reference. Task 6
   vì vậy **không** suy nhãn 0/1 từ winner A/B.

Cách dùng đúng: xem judge là một bộ lọc tự động có thể sai, luôn kiểm chứng các case
lệch (ở đây không có case lệch) và re-check thủ công khi đổi model hoặc đổi prompt.
Với 5 cặp và 10 nhãn, kết quả đủ để chứng minh quy trình chạy đúng và đủ để phát
hiện lỗi phiên bản/phủ định, nhưng chưa đủ để đặt ngưỡng chất lượng dài hạn.

Ghi chú kỹ thuật về môi trường: `.env` của máy chạy để `GEMINI_API_KEY` trống và
trỏ `OPENAI_BASE_URL` sang endpoint OpenAI-compatible khác, nên các số này không
phải của Gemini. Toàn bộ đường dây Gemini đã implement và đã kiểm chứng cấu hình,
nhưng không có lượt gọi Gemini nào được thực hiện.
