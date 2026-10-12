"""m1 — corpus loading and hierarchical chunking (Day 18 interface).

Contract (doc/PIPELINE_INTERFACE.md):
    load_documents()                        -> list[{text: str, metadata: dict}]
    chunk_hierarchical(text, metadata=...)  -> (parents, children)
    child.text / child.metadata / child.parent_id

Parent chunks (≈HIERARCHICAL_PARENT_SIZE) keep the surrounding context that a
reranker would otherwise lose; child chunks (≈HIERARCHICAL_CHILD_SIZE) are what the
vector index retrieves. `setup_answers.py` keeps the parent id on every child and
merges metadata after enrichment.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

import config

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")


@dataclass
class Chunk:
    """A retrievable unit. `parent_id` links back to its parent context."""

    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


@dataclass
class Parent:
    """A large context block owning one or more child chunks."""

    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str = ""
    children: list[Chunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def load_documents(data_dir: str | None = None) -> list[dict]:
    """Load every supported file under `config.DATA_DIR` as `{text, metadata}`.

    Files that yield no extractable text (image-only/scanned PDFs) are reported
    instead of silently dropped: their content is genuinely absent from retrieval,
    which matters when a low RAGAS context_recall is traced back to the corpus.
    """
    root = data_dir or config.DATA_DIR
    documents: list[dict] = []
    empty: list[str] = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        suffix = os.path.splitext(name)[1].lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = _read_pdf(path) if suffix == ".pdf" else _read_text(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the corpus
            print(f"  ⚠️  {name}: unreadable ({type(exc).__name__}: {exc})")
            empty.append(name)
            continue
        text = normalize_text(text)
        if not text:
            empty.append(name)
            continue
        documents.append({
            "text": text,
            "metadata": {
                "source": name,
                "doc_type": suffix.lstrip("."),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            },
        })
    if empty:
        print(f"  ⚠️  {len(empty)} file(s) yielded NO extractable text (image-only PDF or scan); "
              f"their content is absent from retrieval: {empty}")
    return documents


def normalize_text(text: str) -> str:
    """Collapse trailing spaces and runs of blank lines without losing paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+")


def split_units(text: str) -> list[str]:
    """Split into paragraph/sentence units used as chunking material."""
    units: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if len(block) <= config.HIERARCHICAL_CHILD_SIZE:
            units.append(block)
            continue
        units.extend(part.strip() for part in _SENTENCE_SPLIT.split(block) if part.strip())
    return units


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Word-boundary sliding windows of roughly `size` characters."""
    if len(text) <= size:
        return [text]
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= size:
            chunks.append(" ".join(current))
            keep: list[str] = []
            kept = 0
            for previous in reversed(current):
                if kept >= overlap:
                    break
                keep.insert(0, previous)
                kept += len(previous) + 1
            current = keep
            length = kept
    tail = " ".join(current).strip()
    if tail and (not chunks or tail != chunks[-1]):
        chunks.append(tail)
    return chunks


def chunk_hierarchical(text: str, metadata: dict | None = None,
                       parent_size: int | None = None,
                       child_size: int | None = None) -> tuple[list[Parent], list[Chunk]]:
    """Build parent/child chunks. Child order is stable and parent ids are content-derived."""
    metadata = dict(metadata or {})
    parent_size = parent_size or config.HIERARCHICAL_PARENT_SIZE
    child_size = child_size or config.HIERARCHICAL_CHILD_SIZE
    overlap = max(32, child_size // 5)

    units = split_units(text)
    if not units:
        return [], []

    parents: list[Parent] = []
    buffer: list[str] = []
    length = 0
    for unit in units:
        buffer.append(unit)
        length += len(unit) + 2
        if length >= parent_size:
            parents.append(_make_parent("\n\n".join(buffer), metadata, len(parents)))
            buffer, length = [], 0
    if buffer:
        parents.append(_make_parent("\n\n".join(buffer), metadata, len(parents)))

    children: list[Chunk] = []
    for parent in parents:
        for piece in _windows(parent.text, child_size, overlap):
            child = Chunk(
                text=piece,
                metadata={**metadata, "parent_id": parent.parent_id},
                parent_id=parent.parent_id,
            )
            parent.children.append(child)
            children.append(child)
    return parents, children


def _make_parent(text: str, metadata: dict, index: int) -> Parent:
    source = metadata.get("source", "document")
    parent_id = f"{source}#p{index:03d}"
    parent = Parent(text=text, metadata={**metadata, "parent_id": parent_id}, parent_id=parent_id)
    return parent
