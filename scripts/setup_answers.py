"""
Setup script: chạy Day 18 pipeline trên 50 câu hỏi → lưu artifacts/answers_50q.json

Chạy TRƯỚC khi bắt đầu Phase A:
    python scripts/setup_answers.py --live

Yêu cầu:
    1. Có 5 module m1-m5 với giao diện trong doc/PIPELINE_INTERFACE.md
       Script tự lắp module; không gọi src/pipeline.py.
    2. docker compose -f infra/docker-compose.yml up -d  (Qdrant đang chạy trên port 6333)
    3. .env có GEMINI_API_KEY (hoặc OPENAI_API_KEY) và OPENAI_BASE_URL trỏ tới
       endpoint OpenAI-compatible của Gemini:
       https://generativelanguage.googleapis.com/v1beta/openai/

Provider: mọi lời gọi LLM đi qua src/llm_client.py (client dùng chung, retry/backoff
429 + pacing cho free tier). Tiến độ được lưu sau từng câu để lần chạy sau không
trả phí lại các lượt đã thành công.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import src.llm_client as llm
from config import ANSWERS_PATH, GENERATION_MODEL, ROOT_DIR, TEST_SET_PATH
from src.checkpoint import ResumableStore, atomic_write_json

GENERATION_SYSTEM_PROMPT = (
    "Trả lời CHỈ dựa trên context được cung cấp. Nếu context không chứa thông tin, "
    "trả lời đúng câu: 'Không tìm thấy.' Không suy diễn, không thêm kiến thức ngoài context. "
    "Nêu rõ phiên bản chính sách (ví dụ v2023/v2024) khi context có nhắc tới."
)


def check_day18_files() -> bool:
    required = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ROOT_DIR, f))]
    if missing:
        print("\n❌ Thiếu files từ Day 18. Copy chúng vào src/ trước:\n")
        for f in missing:
            print(f"   cp <Day18>/src/{os.path.basename(f)} src/")
        return False
    print(f"✓ Day 18 source files: {len(required)}/{len(required)} found")
    return True


def build_pipeline():
    # This assembler calls m1/m2/m3/m5 directly, NOT src.pipeline.py.
    from src.m1_chunking import load_documents, chunk_hierarchical
    from src.m2_search import HybridSearch
    from src.m3_rerank import CrossEncoderReranker
    from src.m5_enrichment import enrich_chunks
    from config import RERANK_TOP_K

    print("\n[1/3] Chunking + enriching documents...")
    t0 = time.time()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })
    print(f"  ✓ {len(docs)} documents → {len(all_chunks)} child chunks ({time.time()-t0:.1f}s)")

    t_enrich = time.time()
    enriched = enrich_chunks(all_chunks)
    if enriched:
        if len(enriched) != len(all_chunks):
            raise ValueError("enrich_chunks must preserve chunk count/order; see doc/PIPELINE_INTERFACE.md")
        all_chunks = [{"text": e.enriched_text,
                       "metadata": {**original["metadata"], **e.auto_metadata}}
                      for original, e in zip(all_chunks, enriched)]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t_enrich:.1f}s)")
    else:
        print(f"  ✓ Using {len(all_chunks)} raw chunks ({time.time()-t_enrich:.1f}s) "
              "[enrichment disabled, uncredentialed, or previously cached as empty]")

    print("\n[2/3] Indexing (BM25 + Dense)...")
    t0 = time.time()
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed {len(all_chunks)} chunks ({time.time()-t0:.1f}s)")

    print("\n[3/3] Loading reranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)")

    return search, reranker, RERANK_TOP_K


def run_query(q: str, search, reranker, top_k: int, client=None) -> tuple[str, list[str], str, str | None]:
    """Retrieve contexts, then generate an answer with the shared client.

    Returns (answer, contexts, generation_mode, generation_model). Extractive
    fallback is recorded truthfully when there is no key/context or generation
    fails; `generation_model` stays null in that case.
    """
    results = search.search(q)
    docs    = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(q, docs, top_k=top_k)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    if (llm.resolve_api_key() or client is not None) and contexts:
        try:
            if client is None:
                client = llm.get_client(role="generation")
            ctx = "\n\n".join(contexts)
            content, used_model = llm.chat_text_response(
                client, model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nCâu hỏi: {q}"},
                ],
                label=f"generation [{q[:40]}]")
            return content, contexts, "llm", used_model
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}")

    return (contexts[0] if contexts else "Không tìm thấy thông tin."), contexts, "extractive_fallback", None


def _answer_record(item: dict, search, reranker, top_k: int, client) -> dict:
    answer, contexts, generation_mode, generation_model = run_query(
        item["question"], search, reranker, top_k, client)
    return {
        "id":           item["id"],
        "distribution": item["distribution"],
        "question":     item["question"],
        "answer":       answer,
        "contexts":     contexts,
        "ground_truth": item["ground_truth"],
        "generation_mode": generation_mode,
        "generation_model": generation_model,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assemble previous modules and generate 50 answers (API cost).")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only generate the first N questions (smoke runs); 0 = all")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Override LLM_MAX_CONCURRENCY for this run")
    parser.add_argument("--reset-progress", action="store_true",
                        help="Ignore previously generated answers and start over")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Generation/enrichment may call APIs. Pass --live after completing pre-class setup.")

    try:
        llm.require_api_key()
    except llm.MissingCredential as error:
        parser.error(str(error))
    print("=" * 60)
    print("K4 / TRACK 3B SETUP — Generating answers for 50 questions")
    print("=" * 60)
    print(f"Endpoint : {llm.resolve_base_url()}")
    print(f"Model    : {GENERATION_MODEL}")
    print(f"Pacing   : {config.LLM_MIN_INTERVAL_S}s between calls, "
          f"concurrency {args.concurrency or config.LLM_MAX_CONCURRENCY}")

    if not check_day18_files():
        sys.exit(1)

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)
    if args.limit:
        test_set = test_set[:args.limit]
    print(f"✓ Loaded {len(test_set)} questions (factual/multi_hop/adversarial)")

    llm.set_pacing(config.LLM_MIN_INTERVAL_S)
    client = llm.get_client(role="generation")

    try:
        search, reranker, top_k = build_pipeline()
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("→ Đảm bảo bạn đã copy src/ từ Day 18 và đã pip install -r requirements/requirements.txt")
        sys.exit(1)

    progress_path = config.ANSWERS_PROGRESS_PATH
    if args.reset_progress and os.path.exists(progress_path):
        os.remove(progress_path)
    store = ResumableStore(progress_path, key_field="id")
    if store.recovered_from:
        print(f"  ⚠️  Unreadable progress file moved to {store.recovered_from}")
    order = [item["id"] for item in test_set]
    todo = [item for item in test_set if item["id"] not in store]
    print(f"\nRunning {len(test_set)} queries "
          f"({len(test_set) - len(todo)} already completed, {len(todo)} to go)...")

    t_start = time.time()
    workers = max(1, args.concurrency or config.LLM_MAX_CONCURRENCY)
    done = len(test_set) - len(todo)
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_answer_record, item, search, reranker, top_k, client): item
                       for item in todo}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - keep the batch alive
                    print(f"  ⚠️  question {item['id']} failed entirely: "
                          f"{type(exc).__name__}: {exc}")
                    continue
                store.upsert(record)
                done += 1
                print(f"  [{done}/{len(test_set)}] id={record['id']} "
                      f"mode={record['generation_mode']} ({time.time()-t_start:.0f}s elapsed)")

    answers = store.ordered(order)
    missing = [qid for qid in order if qid not in store]
    if missing:
        print(f"\n❌ Missing {len(missing)} answers: {missing}")
        print(f"   Tiến độ đã lưu tại {progress_path}; chạy lại để tiếp tục.")
        sys.exit(1)

    atomic_write_json(ANSWERS_PATH, answers)
    llm_counts = sum(1 for row in answers if row["generation_mode"] == "llm")
    print(f"\n✓ Saved {len(answers)} answers → artifacts/answers_50q.json")
    print(f"  generation: {llm_counts} llm / {len(answers) - llm_counts} extractive_fallback")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print("\n→ Bây giờ bắt đầu Phase A:")
    print("     python scripts/run_experiments.py --phase a --live")


if __name__ == "__main__":
    # Line-buffered so a long live run is observable when stdout is redirected to a file.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    main()
