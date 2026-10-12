"""m3 — cross-encoder reranking of the hybrid candidate list (Day 18 interface).

Contract (doc/PIPELINE_INTERFACE.md):
    CrossEncoderReranker()                    -> instance with .rerank(...)
    rerank(question, docs, top_k)             -> list[object] with .text
    docs = list[{text, score, metadata}]; empty result means "use top-3 search rows"

flashrank runs a small ONNX cross-encoder on CPU: the hybrid retriever maximises
recall over 20 candidates, and the reranker restores precision for the 3 contexts
that actually reach the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config


@dataclass
class RerankResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class CrossEncoderReranker:
    """Lazy flashrank wrapper with a score-order fallback."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.RERANK_MODEL
        self._ranker = None
        self._failed = False

    def _get_ranker(self):
        if self._failed:
            return None
        if self._ranker is None:
            try:
                from flashrank import Ranker
                self._ranker = Ranker(model_name=self.model_name)
            except Exception as exc:  # noqa: BLE001 - reranking is an optimisation
                print(f"  ⚠️  flashrank unavailable ({type(exc).__name__}: {exc}); "
                      "falling back to hybrid score order")
                self._failed = True
                return None
        return self._ranker

    def rerank(self, question: str, docs: list[dict], top_k: int | None = None) -> list[RerankResult]:
        top_k = top_k or config.RERANK_TOP_K
        candidates = [doc for doc in (docs or []) if isinstance(doc, dict) and doc.get("text")]
        if not candidates:
            return []

        ranker = self._get_ranker()
        if ranker is not None:
            try:
                from flashrank import RerankRequest
                passages = [{"id": i, "text": doc["text"], "meta": doc.get("metadata") or {}}
                            for i, doc in enumerate(candidates)]
                ranked = ranker.rerank(RerankRequest(query=question, passages=passages))
                scored = [
                    RerankResult(
                        text=item["text"],
                        score=float(item.get("score", 0.0)),
                        metadata=dict(candidates[int(item["id"])].get("metadata") or {}),
                    )
                    for item in ranked
                ]
                # Guard against off-domain saturation. A cross-encoder trained on other
                # languages scores every Vietnamese passage as "almost perfectly
                # relevant", so its ordering is noise and reordering on it is strictly
                # worse than the fused ranking. Measured on this corpus with
                # ms-marco-MultiBERT-L-12: 20 candidates all landed in
                # [0.996963, 0.999552] and the top-3 became thuong_tet.md /
                # nghi_phep_khong_luong.md while the fused order had the correct
                # nghi_phep_dac_biet.md first. Two saturation signatures are checked:
                # a tiny spread, and a compressed high band.
                if scored:
                    top = max(item.score for item in scored)
                    bottom = min(item.score for item in scored)
                    spread = top - bottom
                    if spread < config.RERANK_MIN_SPREAD or bottom >= config.RERANK_SATURATION_FLOOR:
                        print(f"  ⚠️  reranker saturated on this corpus "
                              f"(min={bottom:.5f} max={top:.5f} spread={spread:.2e}); keeping hybrid order")
                        return [RerankResult(text=doc["text"], score=float(doc.get("score", 0.0)),
                                             metadata=dict(doc.get("metadata") or {}))
                                for doc in scored_by_hybrid(candidates)[:top_k]]
                return scored[:top_k]
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️  flashrank rerank failed ({type(exc).__name__}: {exc}); "
                      "falling back to hybrid score order")

        return [RerankResult(text=doc["text"], score=float(doc.get("score", 0.0)),
                             metadata=dict(doc.get("metadata") or {}))
                for doc in scored_by_hybrid(candidates)[:top_k]]


def scored_by_hybrid(candidates: list[dict]) -> list[dict]:
    """Candidates in the retriever's own score order (stable for equal scores)."""
    return sorted(candidates, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)
