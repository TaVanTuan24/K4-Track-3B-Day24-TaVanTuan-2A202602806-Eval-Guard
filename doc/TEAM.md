# Nhóm thực hiện K4 Track 3B

Tên nhóm: K4-Track-3B-TaVanTuan (bài cá nhân, một thành viên).

Repo chung: https://github.com/TaVanTuan24/K4-Track-3B-Day24-TaVanTuan-2A202602806-Eval-Guard
(GitHub báo repo cũ `K4-Track-3B-TaVanTuan-...` đã chuyển sang địa chỉ này; đây là
địa chỉ chuẩn và đã được set làm `origin`.)

Trưởng nhóm/người nộp: Tạ Văn Tuấn — 2A202602806.

Ngày lab/deadline áp dụng: theo thông báo của lớp — Asia/Ho_Chi_Minh (UTC+07:00).

Commit dùng để chấm: `cd8b86cbaedb19b6df1a109ec18a370cf79733c5` (commit cd8b86c, nhánh
`main`) — commit này chứa toàn bộ source, artifact và báo cáo của bài.

Mỗi nhóm nộp một repo. Nhóm này chỉ có một thành viên nên bảng bên dưới có một dòng
và mục INDIVIDUAL có một mục tương ứng.

## Thành viên và phân công

| STT | Họ và tên | MSSV | Vai trò / công việc thật | Bằng chứng |
|---:|---|---|---|---|
| 1 | Tạ Văn Tuấn | 2A202602806 | Toàn bộ bài: chuyển provider sang client OpenAI-compatible dùng chung; viết m1–m5; Phase A/B/C; guardrails; chạy thật 4 batch | commit của bài nộp; artifacts/answers_50q.json; artifacts/ragas_50q.json; artifacts/judge_results.json; artifacts/guard_results.json |

Người tích hợp/chạy batch/nộp link: Tạ Văn Tuấn — tự viết `src/llm_client.py` và
`src/checkpoint.py`, tự ráp pipeline trong `scripts/setup_answers.py`, chạy
`setup_answers.py --live` rồi `run_experiments.py --phase a/b/c --live`; bằng chứng
là ba file JSON trong `artifacts/` cùng timestamp UTC trong metadata.

## Môi trường và chuẩn bị trước lớp

OS/Python/compiler: Windows 11 (10.0.26200), Python 3.11.15 (uv-managed), venv
`.venv311`. Visual Studio 2022 có trên máy nhưng **thiếu workload C++**, nên không
build được `annoy` từ sdist.

Virtualenv/dependencies/imports: `uv venv .venv311 --python 3.11.15` rồi
`uv pip install -r requirements/requirements.txt -c requirements/constraints-core.txt`.
Import thật đã kiểm chứng: `ragas, nemoguardrails, presidio_analyzer, spacy, torch,
sentence_transformers, flashrank, qdrant_client, langchain_openai` đều OK.
`uv pip check` báo đúng **một** incompatibility: `nemoguardrails requires
annoy>=1.17.3`. Không có wheel Windows cho bất kỳ bản `annoy` nào trên PyPI,
conda-forge không có build win-64, và máy thiếu compiler C++; hệ quả là
`nemoguardrails.embeddings.basic` (AnnoyIndex) không import được. Đây là index của
**dialog rails**, không dùng cho `self check input/output` mà bài bật — bằng chứng
là `scripts/check_api.py --live` chạy đủ 4 quyết định rail đúng.

Models/Qdrant/50 answers đã chuẩn bị: Qdrant `v1.9.7` qua
`infra/docker-compose.yml`, collection `lab24_production` có 110 điểm (dim 1024).
Embedding `BAAI/bge-m3` chạy local; `hf download` ghi file 0 byte trên máy này dù
Hub truy cập được, nên weight được tải bằng `curl` vào `models/bge-m3` (đã
git-ignore) và `src/embeddings.py` ưu tiên thư mục đó. Reranker flashrank
(`ms-marco-MultiBERT-L-12`) tự tải qua HTTPS. `en_core_web_lg 3.8.0` cài từ wheel
local. `torch` phải là 2.8.0: bản 2.14.1 lỗi `WinError 1114` khi import, bản 2.5.1
import được nhưng `transformers` 5.19 cần `torch.accelerator`.
`artifacts/answers_50q.json` có đủ 50 answers, 20/20/10, tất cả
`generation_mode = llm`.

Nếu full NeMo chưa hoàn chỉnh, ghi lỗi thật: xem mục `annoy` ở trên — phần thiếu là
một dependency không cài được, kèm bằng chứng rằng các rail mà bài dùng vẫn chạy.

## INDIVIDUAL

### TaVanTuan-2A202602806

- Phần được giao và phần thực tế đã làm: toàn bộ bài. Cụ thể: `src/llm_client.py`
  (factory client OpenAI-compatible dùng chung + retry/backoff 429/5xx + pacing),
  `src/checkpoint.py` (ghi JSON nguyên tử, resume an toàn), `src/embeddings.py`,
  `src/m1_chunking.py`, `src/m2_search.py`, `src/m3_rerank.py`, `src/m4_eval.py`,
  `src/m5_enrichment.py`, `scripts/setup_answers.py`, `scripts/check_api.py`,
  `scripts/check_evaluator.py`, `guardrails/config.yml` (engine Gemini), và
  Tasks 1–12 trong `src/phase_a_ragas.py`, `src/phase_b_judge.py`,
  `src/phase_c_guard.py`.
- Bằng chứng: ba artifact `ragas_50q.json` (139.544 bytes),
  `judge_results.json` (13.720 bytes), `guard_results.json` (15.076 bytes) cùng
  `answers_50q.json` (128.488 bytes), ba báo cáo trong `doc/reports/`, và commit của
  bài nộp ghi ở mục cuối.
- Cách kiểm tra: offline `python -m pytest tests/ -q` cho **86 passed, 3 deselected**
  (và `-m support` cho 60 passed); live `python scripts/check_api.py --live` cho 12
  check PASS (1 lượt chat, structured JSON, NeMo input chặn jailbreak/cho qua câu
  sạch, NeMo output chặn output nhạy cảm/giữ output sạch);
  `python scripts/check_evaluator.py --live --rows 3` để kiểm tra evaluator trước khi
  chạy đủ 50; `python scripts/run_experiments.py --phase a/b/c --live`.
- Điều đã học: (1) LLM provider trong NeMo 0.10.0 được chọn theo **tên model**, nên
  không thể dùng `engine: openai` với model Gemini mà phải đăng ký engine riêng;
  (2) `max_tokens=3` mặc định của self-check khiến model trả `None` và mọi input bị
  chặn — phải khai báo `max_tokens` trong prompt config; (3) coi mọi entity Presidio
  là PII làm chặn nhầm 8/8 câu sạch, phải lọc theo allowlist; (4) retry/backoff và
  cache theo từng câu là điều kiện để chạy được batch thật trên hạn mức thấp.
- Giới hạn/cách xử lý: `.env` trên máy này để `GEMINI_API_KEY` trống và trỏ
  `OPENAI_BASE_URL` sang endpoint OpenAI-compatible khác
  (`deepseek-ai/DeepSeek-V4.1-Flash`), nên **không có lượt gọi Gemini nào được thực
  hiện** và mọi số đo là của provider đó; đường dây Gemini đã implement và đã kiểm
  chứng cấu hình nhưng cần key Gemini để chạy. Hai PDF chỉ có ảnh nên không trích
  được chữ; reranker bão hoà trên tiếng Việt nên bị chặn đổi thứ tự; latency input
  guard P95 1134 ms vượt target 500 ms và được giữ nguyên là số thật.
- Nguồn/AI hỗ trợ nếu có: dùng AI hỗ trợ viết code, đọc tài liệu và soạn báo cáo;
  mọi kết luận kỹ thuật đều được kiểm chứng lại bằng lệnh chạy thật trên máy (pytest
  offline, `check_api.py`, ba batch `run_experiments.py`) và bằng cách đọc trực tiếp
  source của `nemoguardrails 0.10.0`, `ragas 0.1.22`, `flashrank` đã cài. Không có
  số liệu nào được điền tay: tất cả lấy từ JSON do runner sinh.

## Cập nhật sau khi nộp

Chưa có cập nhật sau deadline. Commit dùng để chấm là
`cd8b86cbaedb19b6df1a109ec18a370cf79733c5` (cd8b86c), đã chứa đủ source, ba artifact
và ba báo cáo. Commit kế tiếp chỉ ghi lại chính SHA này vào TEAM.md, không thay đổi
code, artifact hay nội dung phân tích. Nếu repo được sửa sau deadline thì commit và
thời điểm sửa sẽ được ghi rõ tại đây.
