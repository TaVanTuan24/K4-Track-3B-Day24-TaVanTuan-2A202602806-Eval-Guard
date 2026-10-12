# Môi trường K4 Track 3B

## Phiên bản mục tiêu và cách kiểm tra

Dùng Python 3.11 cho lab này. Tạo virtualenv mới sau khi tải starter, cài
dependencies và chạy `pip check` trên máy của nhóm. Tests hỗ trợ chạy không
cần API hay model; student tests vẫn fail nếu các hàm TODO chưa được làm.

Bộ phiên bản lõi trong starter: RAGAS 0.1.22; LangChain 0.2.17; langchain-core
0.2.43; community 0.2.19; openai integration 0.1.25; OpenAI SDK 1.109.1; NeMo
0.10.0; Annoy 1.17.3; NumPy 1.26.4. Các pins nằm trong `requirements/constraints-core.txt`.
Đây là constraints cho tập lõi, **không phải lock toàn bộ dependencies**.

Trên Windows, build `annoy==1.17.3` cần Microsoft Visual C++ 14.0 trở lên.
Sau khi cài compiler, nhóm chạy các bước kiểm chứng phía dưới trên chính môi
trường sẽ làm lab. Kết quả unit tests hoặc pip resolve không chứng minh full
NeMo đã cài/chạy được.

Theo [README của NeMo v0.10.0](https://github.com/NVIDIA/NeMo-Guardrails/blob/v0.10.0/README.md),
bản này hỗ trợ Python đến 3.11 và cần compiler/dev tools cho Annoy. Không lấy
hướng dẫn bản NeMo mới nhất để suy ra bản 0.10.0 hỗ trợ Python 3.12+.

## Chuẩn bị compiler và virtualenv trước buổi học

Trên Windows, cài Microsoft C++ Build Tools, chọn workload **Desktop development
with C++**, bao gồm MSVC và Windows SDK. Mở lại terminal sau cài; dùng Developer
PowerShell nếu compiler chưa có trong PATH. Làm theo
[hướng dẫn Microsoft](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170).
Việc cài compiler/tải model không thuộc 15 phút kiểm tra đầu buổi.

Từ gốc repo trên PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements/requirements.txt -c requirements/constraints-core.txt
.\.venv\Scripts\python.exe -m pip check
```

Không bắt buộc kích hoạt virtualenv: dùng đường dẫn interpreter như trên.
Nếu muốn chạy `Activate.ps1` và gặp lỗi execution policy, có thể cho phép script
chỉ trong phiên PowerShell hiện tại rồi kích hoạt:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Nếu policy của tổ chức vẫn chặn, dùng interpreter trực tiếp; không thay policy
toàn máy. Xem [hướng dẫn venv Python 3.11](https://docs.python.org/3.11/library/venv.html).
macOS/Linux dùng `python3.11 -m venv .venv` và `source .venv/bin/activate`;
cần toolchain C++ của hệ điều hành khi build Annoy.

## Kiểm chứng cài đặt và tải model

Cài thành công qua pip chưa chứng minh đã tải model hoặc API kết nối được.
Kiểm tra các lớp riêng biệt để biết lỗi nằm ở đâu:

```bash
python -m pip check
python -c "import ragas, nemoguardrails, presidio_analyzer, spacy; print('imports OK')"
python -m spacy download en_core_web_lg
docker compose -f infra/docker-compose.yml up -d
python scripts/setup_answers.py --live
```

Qdrant dùng tag [v1.9.7](https://github.com/qdrant/qdrant/releases/tag/v1.9.7),
không dùng `latest`. Setup sẽ tải embedding BGE-M3/reranker theo module bài
trước; nhóm cần cache/download xong trước lớp. Model/download/indexing có thể
mất lâu tùy máy và mạng. Generation/enrichment có thể gọi API, có chi phí.

Ghi OS, Python, kết quả pip/import/compiler/model download trong TEAM. Nếu full
stack cài chưa xong, ghi lỗi thật; không điền “tương thích” dựa trên dry-run.

## Tạo môi trường unit test offline

Khi chỉ sửa logic/scaffold, không cần cài NeMo, ML model hoặc C++ compiler.
Tạo môi trường riêng với các dependencies đã pin:

```powershell
py -3.11 -m venv .venv-tests
.\.venv-tests\Scripts\python.exe -m pip install -r requirements/requirements-test.lock.txt
.\.venv-tests\Scripts\python.exe -m pip check
.\.venv-tests\Scripts\python.exe -m pytest tests/ -m support -q
```

`requirements/requirements-test.txt` ghi dependencies trực tiếp; file `.lock.txt` khóa
toàn bộ dependencies của tập test. Đây chỉ là lock unit test, không thay constraints hoặc
cài đặt full runtime. Không commit virtualenv, key hoặc cache model.

## Ghi nhận thực tế trên máy chạy bài này

Windows 11 (10.0.26200), Python **3.11.15** (uv-managed). Virtualenv full runtime:
`.venv311/` (đã git-ignore). Lệnh cài và kết quả thật:

```powershell
uv venv .venv311 --python 3.11.15
uv pip install -r requirements/requirements.txt -c requirements/constraints-core.txt
```

Bốn điểm khác starter cần ghi lại vì đây là lỗi thật gặp phải, không phải suy đoán:

1. **`annoy` không cài được trên máy này.** Không có wheel Windows cho bất kỳ bản
   `annoy` nào trên PyPI (chỉ có sdist), conda-forge không có build `win-64` cho
   `annoy`, và Visual Studio 2022 trên máy thiếu workload C++ nên không build được
   sdist. Hệ quả: `pip check` báo `nemoguardrails requires annoy>=1.17.3`, và
   `nemoguardrails.embeddings.basic` (AnnoyIndex) không import được. Đây là index
   dùng cho **dialog rails**, không dùng cho `self check input/output` mà bài này
   bật. Kiểm chứng thực tế: pre-flight `scripts/check_api.py --live` chạy đủ 4 quyết
   định rail (chặn jailbreak, cho qua câu sạch, chặn output nhạy cảm, giữ output
   sạch) — stack guard của Phase C hoạt động. NeMo vẫn khởi tạo được vì
   `from nemoguardrails import RailsConfig, LLMRails` không cần annoy.
2. **`torch` phải là 2.8.0.** Bản 2.14.1 cài từ PyPI lỗi
   `OSError [WinError 1114] ... c10.dll` khi import; 2.5.1 import được nhưng
   `transformers` 5.19 gọi `torch.accelerator` (chỉ có từ torch 2.8) nên
   `sentence-transformers` không import được. `torch==2.8.0` là bản chạy đúng cả hai.
3. **`python -m pip` không tồn tại trong venv uv** — dùng `uv pip ...` cho mọi thao tác
   package.
4. **`hf download` ghi file 0 byte trên máy này** dù Hub truy cập được (kiểm chứng:
   `curl` tải `config.json` của BAAI/bge-m3 trả về đúng 687 byte, `sentencepiece.bpe.model`
   5 MB, `pytorch_model.bin` 2.27 GB). Vì vậy weight bge-m3 được tải bằng `curl` vào
   `models/bge-m3/` (đã git-ignore), và `src/embeddings.py` tự ưu tiên thư mục local đó
   rồi mới fallback về model id HuggingFace. Reranker flashrank cũng tải qua HTTPS
   trực tiếp nên hoạt động bình thường.

Kiểm chứng import của full runtime:

```powershell
.\.venv311\Scripts\python.exe -c "import ragas, nemoguardrails, presidio_analyzer, spacy,
    torch, sentence_transformers, flashrank, qdrant_client, langchain_openai; print('imports OK')"
.\.venv311\Scripts\python.exe -m spacy download en_core_web_lg   # 3.8.0, install từ wheel local
```

## Tái lập byte và độ ổn định đa nền tảng (line endings)

Đây là lỗi tái lập thật đã gặp và đã sửa, không phải cấu hình cho đẹp.

**Triệu chứng.** Ba artifact ghi `metadata.source_sha256` của các file nguồn, và
`src/report_contracts.py` tính lại hash bằng `Path.read_bytes()`. Repo **không có
`.gitattributes`** và `core.autocrlf=true`, nên git lưu blob ở dạng **LF** nhưng
checkout ra working tree **CRLF**. Vì hash được tính trên byte working tree, artifact
ghi hash của bản CRLF:

```text
fixtures/test_set_50q.json   CRLF (đã ghi)  0e87f26e020a32c62ea0864940720618f39600b9d3378193704cf2f86e88446d
                             LF  (blob)     3ad288426e3066e561737b35b66b671e6c411697dffd0d1161cc7b801e5954cc
```

**Hai nguyên nhân, không chỉ một.** Ngoài git, chính code ghi file cũng phụ thuộc nền
tảng: `atomic_write_json` (`src/checkpoint.py`) và `write_json`
(`src/experiment_support.py`) mở file ở text mode, nên Python dịch `\n` thành
`os.linesep` và **ghi ra CRLF trên Windows**. Nghĩa là kể cả có `.gitattributes`, một
lần chạy lại `setup_answers.py` trên Windows vẫn tạo `answers_50q.json` CRLF trên đĩa,
và hash ghi lại sẽ lại không khớp trên Linux. Cả hai phía đều phải sửa.

**Hệ quả.** `python scripts/check_lab.py` xanh trên đúng máy Windows đã sinh artifact
và đỏ trên mọi clone Linux/CI (working tree LF), với lỗi
`Phase A artifact: Stale experiment source: answers_50q.json`.

**Cách sửa.** (1) `.gitattributes` buộc **LF trên mọi nền tảng** (`* text=auto eol=lf`
cộng rule tường minh cho `*.json/*.md/*.py/*.yml/*.yaml/*.txt/*.ini/*.co/*.example` và
`binary` cho `*.pdf/*.bin/*.onnx/*.gguf/*.zip`) rồi chuẩn hoá lại working tree; (2)
ghim `newline="\n"` trong hai hàm ghi JSON để byte sinh ra giống nhau trên Windows,
Linux và CI. Sau đó hash được ghi lại **đúng một lần** cho khớp byte mới:

```powershell
git add .gitattributes
git commit -m "chore(git): them .gitattributes, ep buoc LF"
git rm --cached -r . -q; git reset --hard -q      # ép checkout lại theo attributes
git ls-files --eol                                 # không còn dòng w/crlf nào
.\.venv311\Scripts\python.exe scripts/refresh_metadata_hashes.py --check
.\.venv311\Scripts\python.exe scripts/refresh_metadata_hashes.py --reason "LF normalization"
```

Lưu ý vận hành: `git reset --hard` và `git checkout-index -a -f` **xoá mọi thay đổi
chưa commit**; chỉ chạy khi cây làm việc đã sạch, và commit trước nếu đang có sửa đổi.

**Bảo đảm không đổi kết quả.** Việc ghi lại hash chỉ chạm
`metadata.source_sha256`: `scripts/refresh_metadata_hashes.py` tính hash từ file thật
(không hardcode checksum), từ chối ghi nếu thiếu file nguồn, và giữ nguyên
`created_at`, `run_kind`, `format_version`, model roles, metric, thứ hạng và raw
latency samples. Không chạy lại RAGAS/Judge/Guardrails vì hash không phải là phép đo.
`tests/test_hash_reproducibility.py` khoá các tính chất này, gồm cả việc tái hiện lỗi
cũ (băm một bản CRLF của file nguồn) và kiểm tra hai hàm ghi JSON không sinh `\r\n`.

**Chỉnh sửa chú giải mà không chạy lại API.** `bottom_10[].diagnosis` là chú giải do
Task 3 sinh, không phải số đo; `scripts/refresh_phase_a_annotations.py` tính lại đúng
trường đó từ `per_question` đã lưu, và **từ chối ghi** nếu `rank`/`avg_score`/
`worst_metric`/thành viên bottom 10 sẽ đổi. Cả hai script đều có `--check` để dùng
trong CI mà không sửa file.

