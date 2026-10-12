# Nhóm thực hiện K4 Track 3B

Tên nhóm: K4-Track-3B-TaVanTuan (bài cá nhân, một thành viên).

Repo chung: https://github.com/TaVanTuan24/K4-Track-3B-Day24-TaVanTuan-2A202602806-Eval-Guard
(GitHub báo repo cũ `K4-Track-3B-TaVanTuan-...` đã chuyển sang địa chỉ này; đây là
địa chỉ chuẩn và đã được set làm `origin`.)

Trưởng nhóm/người nộp: Tạ Văn Tuấn — 2A202602806.

Ngày lab/deadline áp dụng: theo thông báo của lớp — Asia/Ho_Chi_Minh (UTC+07:00).

Nhánh nộp bài: `main`.

Các mốc commit, đặt theo nhãn ổn định. Nhánh `main` vẫn có thể nhận commit mới sau thời
điểm ghi tài liệu này, nên không nhãn nào được mô tả là commit cuối cùng:

| Nhãn | Commit | Nội dung |
|---|---|---|
| **Original live experiment commit** | `cd8b86c` | Tạo kết quả thực nghiệm: 5 module RAG, ba phase, `answers_50q.json` và ba artifact, ba báo cáo, TEAM |
| Line-ending / reproducibility | `fbf601a` | Thêm `.gitattributes` (ép LF) để hash không phụ thuộc nền tảng |
| **Final audit commit** | `832bc81` | Sửa hash metadata, cache theo model/prompt, chẩn đoán lỗi theo bằng chứng, sửa báo cáo |
| Ghi nhận commit | `1c6fa8e` | Chỉ sửa TEAM.md để ghi lại mốc `832bc81` |

**Commit tạo kết quả thực nghiệm là `cd8b86c`.** Mọi commit sau đó chỉ sửa code, tài
liệu, test và hash metadata; **không** commit nào chạy lại RAGAS/Judge/Guardrails hay
tạo lại số đo, nên các giá trị trong artifact vẫn là kết quả của lần chạy thật đó.

SHA để chấm: lấy trực tiếp tại thời điểm nộp bằng `git rev-parse HEAD` (và mở đúng
commit đó trên GitHub) thay vì ghi cứng trong file này — repo không tự nhúng SHA của
chính commit đang được tạo.

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

Tái lập byte của tài liệu (audit cuối): repo không có `.gitattributes` và
`core.autocrlf=true`, nên git lưu blob LF nhưng checkout ra working tree CRLF, trong
khi `metadata.source_sha256` băm chính byte working tree; đồng thời
`atomic_write_json`/`write_json` mở file ở text mode nên Python dịch `\n` thành
`os.linesep` và ghi ra CRLF trên Windows. Hệ quả: repo chỉ pass `check_lab.py` trên
máy đã sinh artifact, còn clone trên Linux (working tree LF) thì `validate_metadata`
báo `Stale experiment source`. Đã sửa cả hai phía: thêm `.gitattributes`
(`* text=auto eol=lf` + rule tường minh cho `*.json/*.md/*.py/*.yml/*.txt` và
`binary` cho PDF) rồi chuẩn hoá lại working tree, **và** ghim `newline="\n"` trong
hai hàm ghi JSON để byte sinh ra giống nhau trên mọi nền tảng. Sau đó hash được ghi
lại bằng `scripts/refresh_metadata_hashes.py` (chỉ đổi `metadata.source_sha256`, giữ
nguyên metric, thứ hạng, timestamp, model). Kiểm chứng: `git ls-files --eol` không
còn file `w/crlf` nào, hash LF khớp blob, và `tests/test_hash_reproducibility.py` tái
hiện đúng lỗi cũ (băm một bản CRLF của file nguồn, và ghi JSON rồi kiểm tra không có
`\r\n`).

Nếu full NeMo chưa hoàn chỉnh, ghi lỗi thật: xem mục `annoy` ở trên — phần thiếu là
một dependency không cài được, kèm bằng chứng rằng các rail mà bài dùng vẫn chạy.

## INDIVIDUAL

### TaVanTuan-2A202602806

- Phần được giao và phần thực tế đã làm: toàn bộ bài. Cụ thể: `src/llm_client.py`
  (factory client OpenAI-compatible dùng chung + retry/backoff 429/5xx + pacing),
  `src/checkpoint.py` (ghi JSON nguyên tử LF-only, resume an toàn), `src/cache_keys.py`
  (fingerprint cache theo model/prompt/input), `src/embeddings.py`,
  `src/m1_chunking.py`, `src/m2_search.py`, `src/m3_rerank.py`, `src/m4_eval.py`,
  `src/m5_enrichment.py`, `scripts/setup_answers.py`, `scripts/check_api.py`,
  `scripts/check_evaluator.py`, `scripts/refresh_metadata_hashes.py`,
  `scripts/refresh_phase_a_annotations.py`, `guardrails/config.yml` (engine Gemini),
  `.gitattributes` (LF đa nền tảng) và Tasks 1–12 trong `src/phase_a_ragas.py`,
  `src/phase_b_judge.py`, `src/phase_c_guard.py`.
  Audit cuối bổ sung `tests/test_hash_reproducibility.py`,
  `tests/test_cache_keys.py`, `tests/test_report_numbers.py` và
  `tests/test_phase_a_diagnosis.py`.
- Bằng chứng: ba artifact `ragas_50q.json` (140.0 KB), `judge_results.json` (13.5 KB),
  `guard_results.json` (14.6 KB) cùng `answers_50q.json` (127.8 KB), ba báo cáo trong
  `doc/reports/`, và commit của bài nộp ghi ở mục cuối. Kích thước đo sau khi đã
  chuẩn hoá LF và ghi lại hash.
- Cách kiểm tra: offline `python -m pytest tests/ -q` cho **152 passed, 3 deselected**
  (`-m support` 110 passed, `-m student` 42 passed); live `python scripts/check_api.py
  --live` cho 12 check PASS (1 lượt chat, structured JSON, NeMo input chặn jailbreak/
  cho qua câu sạch, NeMo output chặn output nhạy cảm/giữ output sạch);
  `python scripts/check_evaluator.py --live --rows 3` để kiểm tra evaluator trước khi
  chạy đủ 50; `python scripts/run_experiments.py --phase a/b/c --live`.
  Hậu kiểm offline, không gọi API: `python scripts/refresh_metadata_hashes.py --check`,
  `python scripts/refresh_phase_a_annotations.py --check`,
  `python scripts/check_lab.py --with-tests` (0 failed checks, 26 check).
  CI offline: `.github/workflows/validate.yml` chạy đúng bốn bước trên (thêm
  `compileall`) trên Ubuntu + Python 3.11 với chỉ 4 gói test, không secrets, không gọi
  provider, không tải model và không sửa artifacts.
- Điều đã học: (1) LLM provider trong NeMo 0.10.0 được chọn theo **tên model**, nên
  không thể dùng `engine: openai` với model Gemini mà phải đăng ký engine riêng;
  (2) `max_tokens=3` mặc định của self-check khiến model trả `None` và mọi input bị
  chặn — phải khai báo `max_tokens` trong prompt config; (3) coi mọi entity Presidio
  là PII làm chặn nhầm 8/8 câu sạch, phải lọc theo allowlist; (4) retry/backoff và
  cache theo từng câu là điều kiện để chạy được batch thật trên hạn mức thấp;
  (5) hash ghi trong metadata phải tính trên byte ổn định đa nền tảng — thiếu
  `.gitattributes` thì `core.autocrlf` làm working tree thành CRLF, và ghi JSON ở text
  mode cũng sinh CRLF trên Windows, nên hash chỉ đúng trên máy đã chạy;
  (6) cache phải khoá theo cả model/prompt chứ không chỉ theo input, nếu không nó trả
  lại điểm của model cũ sau khi đổi `EVALUATOR_MODEL`/`JUDGE_MODEL`.
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

Repo được sửa sau bản hoàn thành bài `cd8b86c` trong **hai vòng audit**, tất cả đều
offline (không gọi API, không tạo lại số đo):

**Vòng 1 — reproducibility + cache + chẩn đoán (commit `fbf601a`, `832bc81`):**

- `metadata.source_sha256` trong ba artifact được ghi lại đúng **một lần** cho khớp
  byte LF sau khi chuẩn hoá line endings, kèm sửa hai hàm ghi JSON để không sinh CRLF
  trên Windows; metric, thứ hạng bottom 10, `created_at`, model roles và raw latency
  samples **không đổi** (đã kiểm chứng bằng so sánh JSON).
- Cache khoá thêm theo model/prompt/version (`src/cache_keys.py`).
- `bottom_10[].diagnosis`/`suggested_fix` xuất lại theo phân loại dựa trên bằng chứng;
  `rank`, `avg_score` và `worst_metric` của cả 10 dòng **không đổi**.
- Ba báo cáo Markdown sửa lại; `blueprint.md` sửa kết luận sai về ngưỡng RAGAS của
  nhóm multi_hop.

**Vòng 2 — final polish (chưa commit tại thời điểm ghi mục này):**

- `README.md` viết lại thành tài liệu của repo **đã hoàn thành bài**: tổng quan ba
  phase, tình trạng Tasks 1–13, kiến trúc (phân biệt phần đã implement với phần chỉ
  đề xuất trong blueprint), hướng dẫn kiểm tra offline, cảnh báo khi chạy live, kết quả
  và giới hạn, và phần tái lập. Không còn mô tả trạng thái starter.
- Thêm `.github/workflows/validate.yml`: CI offline trên Ubuntu + Python 3.11, chỉ cài
  4 gói test, không secrets, không gọi provider, không tải model, không khởi động
  Qdrant/Docker, không sửa artifacts. **Chưa có run thật trên GitHub** nên trạng thái
  CI là *chưa kiểm chứng trên hạ tầng GitHub*; cùng chuỗi lệnh đã được chạy lại trong
  một bản clone sạch trên máy này và tất cả đều exit 0.
- Chẩn đoán Task 3 được cải thiện theo review: phân biệt **total abstention** /
  **partial answer** / **missing context evidence** / **missing specific evidence** /
  **abstention despite available evidence** / **version conflict**, và xác định phiên
  bản bằng **header của tài liệu được truy xuất** thay vì tìm chuỗi phiên bản trong
  thân văn bản. Hệ quả: `bottom_10` có 6/10 dòng đổi nhãn; id 44 được phân loại lại
  thành **version conflict** (bản 1.0 được lấy về, chuỗi `v2.0` chỉ là câu “ĐÃ THAY THẾ
  bởi v2.0” trong chính bản cũ), id 33 thành **partial answer**, id 40 thành **missing
  context evidence** dựa trên độ phủ chủ đề. `rank`, `avg_score`, `worst_metric`,
  `per_question`, `per_distribution`, `failure_clusters` và mọi timestamp **không đổi**.
- Thêm `tests/test_phase_a_diagnosis.py` (16 test theo tình huống, dựng từ dữ liệu tổng
  hợp nên không phụ thuộc id câu hỏi) và cập nhật `doc/reports/failure_clusters.md`
  cho khớp artifacts.

Đối chiếu minh bạch giữa các vòng: metric và mọi kết quả thực nghiệm **giữ nguyên**;
chỉ chú giải (`diagnosis`/`suggested_fix`), tài liệu, test và metadata hash được cập
nhật. Không có lượt gọi API nào, nên không có số đo nào được tạo mới trong cả hai vòng.
