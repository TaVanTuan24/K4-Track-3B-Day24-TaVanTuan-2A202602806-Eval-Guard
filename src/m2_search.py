"""m2 — hybrid retrieval: BM25 (lexical) + dense bge-m3 vectors (Qdrant), fused by RRF.

Contract (doc/PIPELINE_INTERFACE.md):
    HybridSearch()            -> instance with .index(chunks) and .search(question)
    index(chunks)             -> chunks: list[{text, metadata}], no return value
    search(question)          -> list[object] with .text / .score / .metadata

Why hybrid: policy questions mix exact identifiers ("v2024", "55 triệu", "PVI")
that lexical search nails, with paraphrases that only dense retrieval finds.
Reciprocal Rank Fusion combines the two ranked lists without needing comparable
score scales.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config

_RRF_K = 60
_TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ỹà-ỹđĐ]+", re.UNICODE)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Diacritic-preserving lowercase tokens; Vietnamese is syllable-separated."""
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


@dataclass
class _Record:
    index: int
    text: str
    metadata: dict


class HybridSearch:
    """BM25 + dense hybrid retriever backed by Qdrant (in-memory fallback)."""

    def __init__(self, collection_name: str | None = None,
                 qdrant_host: str | None = None, qdrant_port: int | None = None):
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.qdrant_host = qdrant_host or config.QDRANT_HOST
        self.qdrant_port = qdrant_port or config.QDRANT_PORT
        self.records: list[_Record] = []
        self._bm25 = None
        self._vectors = None
        self._client = None
        self._dense_ready = False

    # -- indexing ----------------------------------------------------------

    def index(self, chunks: list[dict]) -> None:
        """Build the BM25 index and push dense vectors to Qdrant."""
        self.records = [
            _Record(index=i, text=chunk["text"], metadata=dict(chunk.get("metadata") or {}))
            for i, chunk in enumerate(chunks)
        ]
        if not self.records:
            raise ValueError("HybridSearch.index() received no chunks.")

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([tokenize(record.text) for record in self.records])

        from src.embeddings import encode
        self._vectors = encode([record.text for record in self.records], show_progress=True)
        self._push_to_qdrant()

    def _push_to_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            print(f"  ⚠️  qdrant-client unavailable ({exc}); using in-memory dense search")
            return
        try:
            client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, timeout=30)
            client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=int(self._vectors.shape[1]), distance=Distance.COSINE),
            )
            payload = [
                {"text": record.text, "metadata": record.metadata, "index": record.index}
                for record in self.records
            ]
            points = [
                PointStruct(id=record.index,
                            vector=self._vectors[record.index].tolist(),
                            payload=payload[record.index])
                for record in self.records
            ]
            client.upsert(collection_name=self.collection_name, points=points, wait=True)
            self._client = client
            self._dense_ready = True
            print(f"  ✓ Qdrant collection '{self.collection_name}' upserted "
                  f"({len(points)} points, dim={self._vectors.shape[1]})")
        except Exception as exc:  # noqa: BLE001 - retrieval still works in memory
            print(f"  ⚠️  Qdrant unavailable ({type(exc).__name__}: {exc}); using in-memory dense search")

    # -- search ------------------------------------------------------------

    def _bm25_ranking(self, question: str) -> list[int]:
        if self._bm25 is None:
            raise RuntimeError("HybridSearch.index() must run before search().")
        scores = self._bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order if scores[i] > 0][:config.BM25_TOP_K]

    def _dense_ranking(self, question: str) -> list[int]:
        from src.embeddings import encode
        query_vector = encode([question])[0]
        if self._dense_ready and self._client is not None:
            try:
                hits = self._client.search(collection_name=self.collection_name,
                                           query_vector=query_vector.tolist(),
                                           limit=config.DENSE_TOP_K, with_payload=False)
                return [int(hit.id) for hit in hits]
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️  Qdrant search failed ({exc}); falling back to in-memory vectors")
                self._dense_ready = False
        # In-memory cosine (vectors are already L2-normalised).
        similarity = self._vectors @ query_vector
        order = sorted(range(len(similarity)), key=lambda i: similarity[i], reverse=True)
        return order[:config.DENSE_TOP_K]

    @staticmethod
    def _rrf(rankings: list[list[int]]) -> dict[int, float]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking):
                fused[index] = fused.get(index, 0.0) + 1.0 / (_RRF_K + rank + 1)
        return fused

    def search(self, question: str) -> list[SearchResult]:
        fused = self._rrf([self._bm25_ranking(question), self._dense_ranking(question)])
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:config.HYBRID_TOP_K]
        return [
            SearchResult(text=self.records[index].text, score=round(score, 6),
                         metadata=dict(self.records[index].metadata))
            for index, score in ordered
        ]
