"""m5 — optional LLM enrichment of chunks (Day 18 interface).

Contract (doc/PIPELINE_INTERFACE.md):
    enrich_chunks(chunks) -> list of the SAME length/order as the input
    each item exposes .enriched_text and .auto_metadata
    [] means "fall back to the raw chunks"

Each item prepends a compact auto-generated header (topic / document type /
policy version / summary) so lexical and dense retrieval see the chunk's subject
even when the body only contains table-like fragments.

Cost control: chunks are enriched in batches (ENRICH_BATCH_SIZE per call) and
every result is cached by chunk hash, so a resumed run never re-pays for work that
already succeeded. A failed batch degrades to the raw chunk instead of shifting
positions, and returns [] only when enrichment is disabled or uncredentialed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import config
import src.llm_client as llm
from src.checkpoint import ResumableStore

SYSTEM_PROMPT = (
    "Bạn gán metadata cho các đoạn văn bản chính sách nội bộ của công ty. "
    "Chỉ trả về JSON, không thêm văn bản khác."
)

USER_TEMPLATE = """Với từng đoạn văn bản dưới đây, hãy xác định:
- topic: chủ đề ngắn (ví dụ: nghỉ phép, lương, VPN, bảo hiểm)
- doc_type: loại tài liệu (quy định, quy trình, bảng, hướng dẫn, thông báo)
- version: phiên bản/năm chính sách nếu xuất hiện (ví dụ: v2024); "" nếu không có
- keywords: 2-5 từ khoá tra cứu
- summary: một câu tóm tắt nội dung chính, không suy diễn ngoài văn bản

Đoạn văn bản:
{payload}

Trả đúng JSON theo schema:
{{"items": [{{"i": 0, "topic": "...", "doc_type": "...", "version": "...",
"keywords": ["..."], "summary": "..."}}]}}
Mỗi item phải có "i" đúng bằng số thứ tự đoạn văn bản. Không thêm hoặc bỏ item."""


@dataclass
class EnrichedChunk:
    enriched_text: str
    auto_metadata: dict = field(default_factory=dict)


def _chunk_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_payload(batch: list[dict]) -> str:
    lines = []
    for position, chunk in enumerate(batch):
        body = " ".join((chunk.get("text") or "").split())
        lines.append(f"[{position}] {body}")
    return "\n".join(lines)


def _header(meta: dict) -> str:
    parts = [str(meta.get(key)).strip() for key in ("topic", "doc_type", "version") if meta.get(key)]
    summary = str(meta.get("summary") or "").strip()
    label = " | ".join(parts)
    if label and summary:
        return f"[{label}] {summary}"
    if label:
        return f"[{label}]"
    return summary


def enrich_chunks(chunks: list[dict], *, client: Any | None = None,
                  store: ResumableStore | None = None, batch_size: int | None = None) -> list[EnrichedChunk]:
    """Enrich every chunk; see the module docstring for the fallback contract."""
    if not chunks:
        return []
    if not config.ENRICH_ENABLED:
        print("  ⚠️  ENRICH_ENABLED=0 — m5 enrichment skipped, using raw chunks")
        return []
    if not llm.resolve_api_key() and client is None:
        print("  ⚠️  No API key configured — m5 enrichment skipped, using raw chunks")
        return []

    batch_size = batch_size or max(1, config.ENRICH_BATCH_SIZE)
    store = store or ResumableStore(config.ENRICHMENT_CACHE_PATH, key_field="hash")
    client = client or llm.get_client(role="enrichment")
    resolved: dict[str, dict] = {}

    keys = [_chunk_key(chunk.get("text") or "") for chunk in chunks]
    cached = sum(1 for key in keys if key in store)
    if cached:
        print(f"  ✓ Reusing {cached}/{len(chunks)} cached enrichment records")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        missing = [(offset, chunk) for offset, chunk in enumerate(batch, start)
                   if keys[offset] not in resolved and keys[offset] not in store]
        if not missing:
            continue
        payload_chunks = [chunk for _, chunk in missing]
        try:
            parsed = llm.chat_json(
                client, model=config.JUDGE_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": USER_TEMPLATE.format(
                              payload=_build_payload(payload_chunks))}],
                label=f"m5 enrichment [{start}:{start + len(payload_chunks)}]")
            items = parsed.get("items")
            if not isinstance(items, list) or len(items) != len(payload_chunks):
                raise ValueError(f"expected {len(payload_chunks)} items, got "
                                 f"{len(items) if isinstance(items, list) else type(items).__name__}")
            for local, item in enumerate(items):
                offset = missing[local][0]
                resolved[keys[offset]] = _clean_metadata(item)
        except Exception as exc:  # noqa: BLE001 - degrade one batch, keep alignment
            print(f"  ⚠️  enrichment batch [{start}:{start + len(payload_chunks)}] failed "
                  f"({type(exc).__name__}: {exc}); keeping those chunks raw")
        else:
            # Persist per batch: a run stopped mid-way (quota, Ctrl-C) must not re-pay
            # for the batches that already succeeded.
            for offset, _ in missing:
                store.upsert({"hash": keys[offset], "meta": resolved[keys[offset]]}, save=False)
            store.save()
            print(f"    · enriched batch [{start}:{start + len(payload_chunks)}] "
                  f"({len(store)}/{len(chunks)} chunks cached)")

    results: list[EnrichedChunk] = []
    for offset, chunk in enumerate(chunks):
        meta = resolved.get(keys[offset]) or (store.get(keys[offset]) or {}).get("meta") or {}
        body = chunk.get("text") or ""
        header = _header(meta)
        results.append(EnrichedChunk(enriched_text=f"{header}\n{body}".strip() if header else body,
                                    auto_metadata=meta))
    return results


def _clean_metadata(item: Any) -> dict:
    if not isinstance(item, dict):
        return {}
    meta: dict = {}
    for key in ("topic", "doc_type", "version", "summary"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            meta[key] = value.strip()
    keywords = item.get("keywords")
    if isinstance(keywords, list):
        cleaned = [str(word).strip() for word in keywords if str(word).strip()]
        if cleaned:
            meta["keywords"] = cleaned[:8]
    return meta
