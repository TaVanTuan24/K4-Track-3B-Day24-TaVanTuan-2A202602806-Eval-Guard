"""Cache-key fingerprints for the resumable live runs.

Why a dedicated module
----------------------
The caches exist so a run stopped by quota never re-pays for work already done. That
only stays *correct* if the key covers everything that can change the answer: a cached
RAGAS score produced by model X must not be served after the evaluator switches to
model Y, and a cached judge verdict must not survive a change of the judge model or of
the judge prompt.

Each fingerprint therefore covers:

* Phase A (`ragas_row_key`) — the metric implementation (library vs builtin), the
  evaluator LLM, the embedding model, the implementation/prompt version, and the row
  inputs (question, answer, contexts, ground truth).
* Phase B pairwise (`pairwise_judge_key`) — the judge model, the pairwise prompt
  version, the question, both candidates and the reference.
* Phase B binary (`binary_judge_key`) — the judge model, the binary prompt version,
  the question, the supplied model_answer and the reference.

The API key is deliberately NOT part of any key: rotating a credential must not throw
away completed work, and a fingerprint must never become a place a secret can leak to
(a checkpoint file on disk). `tests/test_cache_keys.py` asserts that.

A "version" string is included rather than hashing the prompt automatically so the
bump is an explicit, reviewable act when a prompt's meaning changes.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

# Bump these when the corresponding prompt/metric definition changes MEANING. A bump
# invalidates the affected cache rows (they are re-computed, never silently reused).
BUILTIN_METRIC_VERSION = "builtin-metrics-v1"
PAIRWISE_PROMPT_VERSION = "pairwise-judge-v1"
BINARY_PROMPT_VERSION = "binary-judge-v1"

_SEP = "\x1e"   # record separator: cannot appear in model ids or Vietnamese/JSON text
_SUB = "\x1f"   # field separator inside a composite field (contexts list)


def fingerprint(*parts: object) -> str:
    """SHA-256 over the parts, in order. `None` is encoded explicitly, not skipped."""
    digest = hashlib.sha256()
    for part in parts:
        if part is None:
            digest.update(b"\x00none")
        elif isinstance(part, (list, tuple)):
            digest.update(_SUB.join(str(item) for item in part).encode("utf-8"))
        else:
            digest.update(str(part).encode("utf-8"))
        digest.update(_SEP.encode("utf-8"))
    return digest.hexdigest()


def ragas_row_key(*, implementation: str, evaluator_model: str, embedding_model: str,
                  metric_version: str, question: str, answer: str,
                  contexts: Sequence[str], ground_truth: str) -> str:
    """Cache key for one Phase A metric row."""
    return fingerprint("ragas-row-v2", implementation, evaluator_model, embedding_model,
                       metric_version, question, answer, list(contexts), ground_truth)


def pairwise_judge_key(*, judge_model: str, prompt_version: str, question: str,
                       answer_a: str, answer_b: str, reference: str) -> str:
    """Cache key for one swap-and-average pair (both passes travel together)."""
    return fingerprint("pairwise-v2", judge_model, prompt_version, question,
                       answer_a, answer_b, reference)


def binary_judge_key(*, judge_model: str, prompt_version: str, question: str,
                     model_answer: str, reference: str) -> str:
    """Cache key for one 0/1 calibration verdict."""
    return fingerprint("binary-v2", judge_model, prompt_version, question,
                       model_answer, reference)


def assert_no_secret(secret: str | None, *materials: Iterable[str]) -> None:
    """Guard used by tests: no API key may ever be folded into a cache fingerprint."""
    if not secret:
        return
    for material in materials:
        text = _SUB.join(str(item) for item in material)
        if secret in text:
            raise AssertionError("A credential leaked into cache-key material.")
