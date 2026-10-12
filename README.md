# K4 Track 3B — Production Eval + Guardrail Stack (Day 24)

Repo bài nộp: [TaVanTuan24/K4-Track-3B-Day24-TaVanTuan-2A202602806-Eval-Guard](https://github.com/TaVanTuan24/K4-Track-3B-Day24-TaVanTuan-2A202602806-Eval-Guard)
— nhánh `main`, bài cá nhân (xem [doc/TEAM.md](doc/TEAM.md)).

**Trạng thái: bài đã hoàn thành và đã chạy thật.** Bốn artifact trong `artifacts/` là
kết quả của các lượt gọi API thật, không phải mock và không phải số điền tay.
Mọi thứ dưới đây mô tả **repo đã hoàn thành**, không còn trạng thái starter
(không còn `# TODO` trong ba phase, `artifacts/` đã có đủ bốn file, TEAM và ba báo
cáo đã điền).

---

## A. Tổng quan dự án

Đánh giá đầu ra một pipeline RAG nội bộ (chính sách nhân sự + CNTT) trên ba phase:

| Phase | Nội dung | Đầu ra |
|---|---|---|
| **A — RAGAS Evaluation** | Bốn metric `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` trên 50 câu thuộc 3 nhóm (factual 20 / multi_hop 20 / adversarial 10); bottom 10; matrix worst_metric × distribution | [`artifacts/ragas_50q.json`](artifacts/ragas_50q.json), [`doc/reports/failure_clusters.md`](doc/reports/failure_clusters.md) |
| **B — LLM-as-a-Judge** | Pairwise A/B theo 3 tiêu chí có hoán đổi thứ tự (swap-and-average), nhãn nhị phân 0/1, Cohen's κ và phân tích position/verbosity bias | [`artifacts/judge_results.json`](artifacts/judge_results.json), [`doc/reports/bias_report.md`](doc/reports/bias_report.md) |
| **C — Guardrails** | Presidio PII + NeMo Guardrails cho input/output, 20 adversarial + 8 clean + 4 output case, và đo P95 latency hai tầng input | [`artifacts/guard_results.json`](artifacts/guard_results.json), [`doc/reports/blueprint.md`](doc/reports/blueprint.md) |

Đầu vào chung là 50 câu hỏi trong [`fixtures/test_set_50q.json`](fixtures/test_set_50q.json)
và [`artifacts/answers_50q.json`](artifacts/answers_50q.json) sinh từ pipeline RAG
5 module (`m1`–`m5`).

---

## B. Tình trạng triển khai hiện tại

### Tasks 1–13

| Task | Nội dung | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1–2 | `group_by_distribution`, `run_ragas_50q` (gọi `m4.evaluate_ragas`, trả `RagasEvaluation` + `evaluator_models` thật) | Hoàn thành | [`src/phase_a_ragas.py`](src/phase_a_ragas.py), 50 dòng trong artifact |
| 3 | `bottom_10` — 10 câu yếu nhất, `worst_metric` là argmin, `diagnosis`/`suggested_fix` dựa trên bằng chứng | Hoàn thành | `bottom_10` của artifact + [failure_clusters.md](doc/reports/failure_clusters.md) |
| 4 | `cluster_analysis` — matrix worst_metric × distribution, nhóm yếu nhất theo mean thấp nhất | Hoàn thành | `failure_clusters` của artifact |
| 5–6 | `pairwise_judge`, `swap_and_average` (2 lượt/cặp, quy về A/B gốc) | Hoàn thành | 5 dòng `pairwise_results` |
| 7 | `cohen_kappa` (công thức thủ công, không phụ thuộc sklearn) | Hoàn thành | κ = 1.0 trên 10 nhãn |
| 8 | `bias_report` — position bias + verbosity + `verbosity_details` | Hoàn thành | `bias_report` của artifact |
| 9a | `pii_scan` — Presidio + recognizer VN (CCCD/CMND, SĐT VN), lọc theo allowlist PII | Hoàn thành | [phase_c_guard.py](src/phase_c_guard.py), kết quả trong guard artifact |
| 9b | `check_input_rail` async — quyết định đọc từ `activated_rails`, không đoán theo từ khoá | Hoàn thành | 16/20 attack bị chặn ở tầng NeMo input |
| 10 | `run_adversarial_suite` — 20 attack + 8 clean, so với `expected` | Hoàn thành | `adversarial` = 20/20, `clean` = 8/8 |
| 11 | `check_output_rail` — output sạch giữ nguyên, output nhạy cảm thay bằng từ chối | Hoàn thành | 4/4 output case |
| 12 | `measure_p95_latency` — scope `input_layers_sequential`, 2 warm-up thật, ≥20 mẫu timed, percentile nội suy tuyến tính | Hoàn thành | 20 mẫu hợp lệ, P95 = **1133.99721 ms** |
| 13 | Blueprint CI/CD (đề xuất) | Hoàn thành | [blueprint.md](doc/reports/blueprint.md) |

### Artifacts và báo cáo

| Nội dung | Đường dẫn | Trạng thái |
|---|---|---|
| 50 answers (answer/contexts/ground_truth tách biệt, `generation_mode`/`generation_model`) | [`artifacts/answers_50q.json`](artifacts/answers_50q.json) | 50 bản ghi, 20/20/10, **50/50 `generation_mode = llm`** (không có extractive_fallback) |
| RAGAS 50 câu + distribution + matrix + bottom 10 | [`artifacts/ragas_50q.json`](artifacts/ragas_50q.json) | 50 dòng, 4 metric hữu hạn trong [0,1] |
| Pairwise + nhãn 0/1 + κ + bias | [`artifacts/judge_results.json`](artifacts/judge_results.json) | 5 cặp, 10 nhãn, κ = 1.0 |
| Guard + latency | [`artifacts/guard_results.json`](artifacts/guard_results.json) | 20 attack, 8 clean, 4 output, 20 mẫu latency |
| Báo cáo A/B/C, TEAM, môi trường, giao diện pipeline, schema | `doc/reports/*.md`, [`doc/TEAM.md`](doc/TEAM.md), [`doc/ENVIRONMENT.md`](doc/ENVIRONMENT.md), [`doc/PIPELINE_INTERFACE.md`](doc/PIPELINE_INTERFACE.md), [`doc/REPORT_SCHEMAS.md`](doc/REPORT_SCHEMAS.md) | Đã điền, không còn placeholder |

### Model thực sự dùng để tạo artifacts

```text
generation : deepseek-ai/DeepSeek-V4.1-Flash
evaluator  : BAAI/bge-m3 (embedding, chạy local) + deepseek-ai/DeepSeek-V4.1-Flash (LLM)
judge      : deepseek-ai/DeepSeek-V4.1-Flash
guard      : deepseek-ai/DeepSeek-V4.1-Flash (NeMo self-check rails)
```

Các lượt gọi đi qua endpoint **OpenAI-compatible** cấu hình trong `.env`
(`OPENAI_BASE_URL`), không phải Google Gemini. `config.py` và
[`.env.example`](.env.example) đặt **Gemini làm mặc định** (`engine: gemini` cho NeMo,
`gemini-3.5-flash-lite` cho model), code đã implement và đã kiểm chứng cấu hình
Gemini, nhưng **không có lượt gọi Gemini nào được thực hiện** trong repo này. Mọi số
đo vì vậy là của endpoint thực tế đã dùng, và **không được mô tả là kết quả Gemini**.

### Giới hạn đã biết

Phân loại rõ **bắt buộc theo rubric** (không chặn nộp) và **giới hạn môi trường**:

| Giới hạn | Loại | Ảnh hưởng |
|---|---|---|
| **P95 input guard 1133.99721 ms > target 500 ms** | Chất lượng, báo cáo trung thực | Không đạt mục tiêu latency. Nguyên nhân: mỗi quyết định NeMo input là một lượt gọi LLM qua mạng (P95 1115.69 ms), còn Presidio local chỉ 23.48 ms. Số đo được giữ nguyên, **không** chỉnh cho đạt |
| **Không chạy Gemini** | Môi trường | `.env` để `GEMINI_API_KEY` trống và trỏ `OPENAI_BASE_URL` sang endpoint khác; cần key Gemini để chạy lại bằng Gemini |
| **`annoy` không cài được** | Môi trường | Không có wheel Windows cho bất kỳ bản `annoy` trên PyPI, conda-forge không có build `win-64`, máy thiếu workload C++ của MSVC. Hệ quả: `nemoguardrails.embeddings.basic` (AnnoyIndex, dùng cho **dialog rails**) không import được. Các rail bài này dùng (`self check input`/`self check output`) vẫn hoạt động — kiểm chứng bằng `scripts/check_api.py --live` |
| **2 PDF chỉ có ảnh** | Dữ liệu | `data/BCTC.pdf` và `data/Nghi_dinh_so_13-2023_...pdf` không trích được chữ, nên nội dung đó không thể vào retrieval. Đã log cảnh báo khi load corpus, không xoá file |
| **Reranker bão hoà trên tiếng Việt** | Chất lượng | `flashrank` `ms-marco-MultiBERT-L-12` cho điểm gần như giống nhau cho mọi đoạn (20 candidate trong 0.996963–0.999552) và đảo lộn thứ tự đúng, nên `src/m3_rerank.py` phát hiện bão hoà và **giữ thứ tự hybrid** |
| **κ = 1.0 trên 10 nhãn** | Mẫu nhỏ | Không nên đọc là judge hoàn hảo: 10 nhãn rất ít, và judge có reference nên κ đo "tái lập phán đoán theo reference", không đo khả năng chấm khi thiếu đáp án |
| **Retrieval là nút cổ chai** | Chất lượng | 21/50 câu thiếu ít nhất một giá trị đặc trưng của reference trong context; nhóm yếu nhất là adversarial (mean 0.6278) |

---

## C. Kiến trúc

```text
Input
  -> Presidio PII            (lọc theo allowlist: VN_CCCD, VN_PHONE, EMAIL_ADDRESS, PHONE_NUMBER, ...)
  -> NeMo input guard        (self check input; quyết định đọc từ activated_rails)
  -> Hybrid RAG              (BM25 + dense bge-m3 qua Qdrant, hợp nhất bằng Reciprocal Rank Fusion)
  -> Reranking               (flashrank cross-encoder, có guard chống bão hoà)
  -> LLM generation          (client OpenAI-compatible dùng chung)
  -> NeMo output guard       (self check output; output nhạy cảm bị thay bằng từ chối)
  -> Response
```

**Đã implement và có bằng chứng chạy thật:** Presidio (`pii_scan`), NeMo input/output
rail (`check_input_rail`, `check_output_rail`), hybrid BM25 + dense + RRF
([`src/m2_search.py`](src/m2_search.py)), rerank có guard bão hoà
([`src/m3_rerank.py`](src/m3_rerank.py)), generation qua client dùng chung
([`src/llm_client.py`](src/llm_client.py)), và đo latency hai tầng input
([`src/phase_c_guard.py`](src/phase_c_guard.py)).

**Chỉ là đề xuất trong blueprint (chưa triển khai):** GitHub Actions CI gate cho RAGAS
và full-guard latency, monitoring/alerting, và mọi mục ghi **“chưa đo”** trong
[blueprint.md](doc/reports/blueprint.md) — gồm latency **RAG**, **NeMo output** và
**full guard có short-circuit**. Latency `input_guard_ms` **không** phải latency của
toàn bộ request. Workflow CI offline trong `.github/workflows/validate.yml` chỉ kiểm
tra tính toàn vẹn (tests, hash, schema), **không** phải quality gate cho RAGAS.

---

## D. Cài đặt và kiểm tra offline (không cần API key)

Ai chấm cũng có thể kiểm tra bài **hoàn toàn offline**: không cần key, không gọi
provider, không tải model embedding, không cần Qdrant/Docker.

**Yêu cầu tối thiểu:** Python **3.11** (repo phát triển trên 3.11.15) và 4 gói trong
[`requirements/requirements-test.lock.txt`](requirements/requirements-test.lock.txt)
(`pytest`, `PyYAML`, `python-dotenv`, `jsonschema`). Đã kiểm chứng các gói này có wheel
cho Linux `cp311` (hoặc là pure-python), nên CI trên Ubuntu không cần compiler.

```bash
git clone https://github.com/TaVanTuan24/K4-Track-3B-Day24-TaVanTuan-2A202602806-Eval-Guard.git
cd K4-Track-3B-Day24-TaVanTuan-2A202602806-Eval-Guard

python -m venv .venv-tests
. .venv-tests/bin/activate                     # Windows: .venv-tests\Scripts\activate
pip install -r requirements/requirements-test.lock.txt

python -m pytest tests/ -q                     # 152 passed, 3 deselected
python scripts/refresh_metadata_hashes.py --check        # source hashes khớp file
python scripts/refresh_phase_a_annotations.py --check    # diagnosis khớp Task 3
python scripts/check_lab.py --with-tests                 # 0 failed checks
```

Ghi chú về các lệnh:

- `pytest tests/ -q` — toàn bộ test offline; socket bị chặn trong `tests/conftest.py`
  nên không thể gọi mạng/API. Marker: `-m support` (110 test) kiểm tra phần scaffold +
  tính toàn vẹn, `-m student` (42 test) kiểm tra hợp đồng các hàm học viên implement.
- `refresh_metadata_hashes.py --check` — recompute SHA-256 của 6 file nguồn và so với
  `metadata.source_sha256` trong ba artifact. Sai hash ⇒ exit 1 (báo report stale).
- `refresh_phase_a_annotations.py --check` — tính lại `bottom_10` từ `per_question` đã
  lưu và so với chú giải trong artifact; chỉ đọc, không sửa file.
  Hai lệnh này **chỉ đọc metadata và không gọi API**.
- `check_lab.py --with-tests` — validate 4 JSON Schema, đối chiếu nguồn/phép tính và
  chạy pytest một lần. Chạy `python scripts/check_lab.py` nếu chỉ muốn kiểm artifacts.
- Integration test (tùy chọn, **có phát sinh chi phí**):
  `python -m pytest -m integration --run-live`.

Môi trường đầy đủ để **chạy live** (NeMo, RAGAS, Presidio/spaCy, Qdrant, bge-m3) xem
[`doc/ENVIRONMENT.md`](doc/ENVIRONMENT.md) và
[`requirements/requirements.txt`](requirements/requirements.txt).

---

## E. Chạy live experiment

> ⚠️ **Cảnh báo:** các lệnh dưới đây gọi API thật, **có thể phát sinh chi phí/quota**,
> và cần key hợp lệ. **Artifacts đã có sẵn trong repo — không cần chạy lại để kiểm tra
> bài.** Việc chấm điểm dựa trên artifacts đã commit.

```bash
# 1) Cấu hình: chép .env.example -> .env rồi điền key
copy .env.example .env            # Windows; Linux: cp .env.example .env
#    GEMINI_API_KEY=...           (hoặc OPENAI_API_KEY=... cho provider khác)
#    OPENAI_BASE_URL=...          (mặc định là endpoint OpenAI-compatible của Gemini)

# 2) Pre-flight: 1 lượt chat, structured JSON, và NeMo input/output rails
python scripts/check_api.py --live --list-models

# 3) (tùy chọn) pre-flight evaluator trước khi chạy đủ 50 câu
python scripts/check_evaluator.py --live --rows 3

# 4) Sinh answers rồi chạy ba phase
python scripts/setup_answers.py --live
python scripts/run_experiments.py --phase a --live
python scripts/run_experiments.py --phase b --live
python scripts/run_experiments.py --phase c --live --latency-samples 20
```

Lưu ý khi đổi provider:

- **Có thể đổi sang bất kỳ endpoint OpenAI-compatible nào** bằng `OPENAI_BASE_URL` +
  key; toàn bộ repo dùng chung một client trong [`src/llm_client.py`](src/llm_client.py),
  nên chỉ cần sửa `.env`.
- **Gemini là cấu hình mặc định, không phải provider đã tạo artifacts.** Muốn chạy
  đúng như artifact hiện tại thì đặt `OPENAI_BASE_URL` của provider tương ứng và
  `GENERATION_MODEL`/`JUDGE_MODEL` = `deepseek-ai/DeepSeek-V4.1-Flash`.
- Nếu tài khoản không có `gemini-3.5-flash-lite`, dùng `check_api.py --list-models` để
  xem model thật rồi đổi trong `.env`.
- `.env` đã bị git-ignore. **Không commit key** vào source, log, ảnh chụp hay report.
- Free tier được điều tiết bằng `LLM_MAX_CONCURRENCY`, `LLM_MIN_INTERVAL_S` và
  retry/backoff lũy tiến (có jitter, tôn trọng `Retry-After`) cho HTTP 429/5xx.
- Tiến độ được lưu an toàn trong `artifacts/cache/` (đã git-ignore): chạy lại không trả
  phí cho các lượt đã thành công, và cache tự invalidate khi đổi model/prompt/input.

---

## F. Kết quả và giới hạn

Số liệu lấy trực tiếp từ artifacts (run timestamp UTC ghi trong `metadata.created_at`
của từng file).

### Phase A — RAGAS (50 câu)

| Distribution | faithfulness | answer_relevancy | context_precision | context_recall | avg_score |
|---|---:|---:|---:|---:|---:|
| factual (20) | 0.9250 | 0.8237 | 0.8417 | 0.7833 | **0.8434** |
| multi_hop (20) | 0.6295 | 0.6111 | 0.7833 | 0.6250 | **0.6623** |
| adversarial (10) | 0.7667 | 0.5944 | 0.7667 | **0.3833** | **0.6278** |
| toàn tập (50) | 0.7751 | 0.6928 | 0.8033 | 0.6400 | **0.7278** |

Nhóm yếu nhất theo mean thấp nhất là **adversarial (0.6278)**, chủ yếu do
`context_recall` 0.3833: bằng chứng hiện hành không được truy xuất đủ (bẫy phiên bản và
phủ định). Metric bị điểm thấp nhất nhiều nhất trên toàn bộ 50 câu là
`answer_relevancy` (16 câu). Chi tiết bottom 10 và nguyên nhân từng nhóm:
[failure_clusters.md](doc/reports/failure_clusters.md).

### Phase B — LLM-as-a-Judge

- 5 cặp A/B (ids 1, 5, 12, 21, 41), nguồn cố định A = answer pipeline,
  B = `model_answer` trong human labels, reference = ground truth.
- Position bias: **1/5 = 0.2** (một cặp không nhất quán khi hoán đổi thứ tự).
- Verbosity: **0.75** trong 4 case decisive (đọc nội dung cho thấy thắng vì đủ ý, không
  phải vì dài).
- 10 nhãn nhị phân: **κ = 1.0**, 10/10 khớp nhãn người — kèm cảnh báo mẫu nhỏ ở mục
  giới hạn. Chi tiết: [bias_report.md](doc/reports/bias_report.md).

### Phase C — Guardrails và latency

| Chỉ số | Kết quả |
|---|---|
| Adversarial | **20/20** attack bị chặn (4 ở tầng Presidio, 16 ở NeMo input) |
| Clean controls | **8/8** được cho qua, `false_positive_rate = 0.0` |
| Output cases | **4/4** (2 safe giữ nguyên, 2 unsafe bị thay bằng từ chối) |
| Latency samples | 20 mẫu timed + 2 warm-up thật, percentile nội suy tuyến tính |
| Presidio P50/P95/P99 | 19.26 / 23.48 / 23.57 ms |
| NeMo input P50/P95/P99 | 773.33 / **1115.69** / 1675.56 ms |
| **Input guard sequential P95** | **1133.99721 ms** — **vượt target 500 ms** |
| `latency_budget_ok` | `false` |

**Không tuyên bố đạt target 500 ms.** Số đo vượt ngưỡng được ghi nguyên và phân tích
trong [blueprint.md](doc/reports/blueprint.md); nút cổ chai là tầng gọi mạng của NeMo.

---

## G. Tái lập (reproducibility)

- **`.gitattributes` ép LF trên mọi nền tảng** (`* text=auto eol=lf` + rule tường minh,
  `binary` cho PDF). Trước đây `core.autocrlf=true` làm working tree thành CRLF trong
  khi blob là LF, nên hash tính trên byte working tree chỉ đúng trên máy đã chạy và
  `check_lab.py` đỏ khi clone trên Linux. Nhớ rằng `hf download` kiểu CLI cũng như bất
  kỳ công cụ nào ghi file đều có thể tạo CRLF; repo đã chuẩn hoá và khoá bằng test.
- **Hash được kiểm tra trên bytes thật.** `metadata.source_sha256` tính bằng
  `Path.read_bytes()`, và `refresh_metadata_hashes.py --check` recompute lại rồi so
  sánh; sửa một ký tự trong file nguồn là report bị coi là stale. Hash **không** hardcode.
- **Hai hàm ghi JSON ghim `newline="\n"`** ([`src/checkpoint.py`](src/checkpoint.py),
  [`src/experiment_support.py`](src/experiment_support.py)) để file sinh ra giống nhau
  trên Windows/Linux/CI.
- **Cache invalidation theo cấu hình.** [`src/cache_keys.py`](src/cache_keys.py) tạo
  fingerprint gồm implementation + evaluator/embedding model + metric version + input
  (Phase A) và judge model + prompt version + input (Phase B), nên đổi model/prompt là
  cache miss chứ không tái sử dụng điểm cũ. API key **không** nằm trong fingerprint.
- **Các script bảo trì chỉ đọc metadata và không gọi API**:
  `scripts/refresh_metadata_hashes.py`, `scripts/refresh_phase_a_annotations.py` (cả hai
  có `--check`, và từ chối ghi nếu thứ hạng/metric sẽ thay đổi).
- **CI offline** (`.github/workflows/validate.yml`) chỉ chạy tests, kiểm tra hash, kiểm
  tra annotation và submission checker trên Ubuntu + Python 3.11: không dùng secrets,
  không gọi provider, không tải model, không khởi động Qdrant/Docker, và không sửa
  artifacts.

---

## Cấu trúc repo

```text
+-- config.py                 # cấu hình tập trung (endpoint, model, pacing/retry, paths)
+-- src/
|   +-- llm_client.py         # client OpenAI-compatible dùng chung + retry/backoff + pacing
|   +-- cache_keys.py         # fingerprint cache theo model/prompt/input
|   +-- checkpoint.py         # ghi JSON nguyên tử (LF), resume an toàn
|   +-- embeddings.py         # bge-m3 local (dùng cho m2 và evaluator m4)
|   +-- m1_chunking.py ... m5_enrichment.py    # 5 module pipeline RAG (Day 18 interface)
|   +-- phase_a_ragas.py      # Tasks 1-4
|   +-- phase_b_judge.py      # Tasks 5-8
|   +-- phase_c_guard.py      # Tasks 9a-12
|   +-- experiment_support.py, experiment_runners.py, report_contracts.py   # hỗ trợ
+-- scripts/
|   +-- setup_answers.py      # sinh 50 answers (live)
|   +-- run_experiments.py    # chạy phase A/B/C (live)
|   +-- check_api.py          # pre-flight API + structured JSON + NeMo rails (live)
|   +-- check_evaluator.py    # pre-flight evaluator Phase A (live, vài dòng)
|   +-- check_lab.py          # submission checker (offline)
|   +-- refresh_metadata_hashes.py, refresh_phase_a_annotations.py   # bảo trì (offline)
+-- tests/                    # test offline (mock, socket bị chặn)
+-- fixtures/  data/  guardrails/  infra/  requirements/
+-- doc/                      # đề bài, rubric, TEAM, ENVIRONMENT, schema, reports
+-- artifacts/                # answers + 3 kết quả + cache (cache đã git-ignore)
+-- .github/workflows/validate.yml    # CI offline
```

---

## Yêu cầu nộp bài (giữ nguyên từ đề bài)

- **Bài nhóm, một repo chung của trưởng nhóm, một link nộp.** Repo này là bài của một
  thành viên; khai báo thật trong [`doc/TEAM.md`](doc/TEAM.md).
- Deadline mặc định **23:59 ngày làm lab, Asia/Ho_Chi_Minh (UTC+07:00)**; áp dụng điều
  chỉnh của key coach trong 48 giờ sau lab theo quy ước chung
  ([doc/RULES.md](doc/RULES.md), [doc/SUBMISSION.md](doc/SUBMISSION.md)).
- Điểm: **100 điểm bắt buộc + tối đa 10 bonus**; tiêu chí chi tiết trong
  [doc/RUBRIC.md](doc/RUBRIC.md). Số checks đạt của checker **không phải** điểm rubric.
- Đọc thêm: [ASSIGNMENT](doc/ASSIGNMENT.md), [CHECKPOINTS](doc/CHECKPOINTS.md),
  [RULES](doc/RULES.md), [REPORT_SCHEMAS](doc/REPORT_SCHEMAS.md),
  [PIPELINE_INTERFACE](doc/PIPELINE_INTERFACE.md).
- Blueprint là **đề xuất** CI/CD, không phải workflow đã chạy trên GitHub.
