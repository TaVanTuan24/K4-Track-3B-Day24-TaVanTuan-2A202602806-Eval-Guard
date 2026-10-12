"""m4 — the four Phase A metrics (Day 18 interface), on the Gemini free tier.

Contract (doc/PIPELINE_INTERFACE.md):
    evaluate_ragas(questions, answers, contexts, ground_truths) -> dict
    per_question[i] exposes faithfulness / answer_relevancy / context_precision /
    context_recall as finite Python floats in [0, 1]
    metadata.evaluator_models lists the real LLM/embedding model ids used

Two real implementations of the same four metrics
-------------------------------------------------
* ``library`` — ragas 0.1.22 with a Gemini-backed ``ChatOpenAI`` and the local
  bge-m3 embeddings.
* ``builtin`` — the same metric definitions implemented directly on the shared
  OpenAI-compatible client. Used when the installed ragas/langchain pair cannot
  talk to the endpoint, or when forced with ``RAGAS_IMPL=builtin``.

Both paths make real API calls; neither invents a score. ``RAGAS_IMPL=auto``
starts with the library and only falls back after a genuine failure, and the path
actually used is recorded in ``metadata.metric_implementation``.

Cost control / resumability
---------------------------
Work is done in batches with a per-question cache keyed by (implementation, input
hash). Library batches are bisected on failure, so one uncooperative question
costs its own retries instead of the whole run, and a run that stops on quota can
be restarted without re-paying for rows already stored. A metric that is still
missing after retries raises with the offending question ids: a partial report is
never passed off as complete.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import config
import src.llm_client as llm
from src.checkpoint import ResumableStore

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
LIBRARY = "library"
BUILTIN = "builtin"
_BATCH = 8

# ---------------------------------------------------------------------------
# Prompt building blocks for the built-in implementation
# ---------------------------------------------------------------------------

_JSON_ONLY = "Chỉ trả về JSON hợp lệ, không thêm văn bản khác."

CLAIM_PROMPT = """Trích xuất mọi khẳng định thực tế trong văn bản sau thành các câu độc lập, ngắn gọn.
Chỉ lấy thông tin xuất hiện trong văn bản, không suy diễn thêm.
Văn bản: \"\"\"{text}\"\"\"
Trả JSON: {{"statements": ["khẳng định 1", "khẳng định 2"]}}
Nếu văn bản không có khẳng định thực tế nào, trả {{"statements": []}}."""

SUPPORT_PROMPT = """Kiểm tra một khẳng định có được CONTEXT hỗ trợ trực tiếp hay không.
CONTEXT: \"\"\"{context}\"\"\"
Khẳng định: \"\"\"{statement}\"\"\"
Trả JSON: {{"verdict": "yes" hoặc "no", "reason": "ngắn gọn"}}
"yes" chỉ khi CONTEXT nêu trực tiếp hoặc suy ra chắc chắn được khẳng định."""

RELEVANCY_PROMPT = """Viết lại thành một câu hỏi mà câu trả lời sau trả lời trực tiếp.
Câu trả lời: \"\"\"{answer}\"\"\"
Trả JSON: {{"question": "câu hỏi"}}"""

PRECISION_PROMPT = """Câu hỏi: \"\"\"{question}\"\"\"
Đáp án tham chiếu cần đạt: \"\"\"{ground_truth}\"\"\"
CONTEXT: \"\"\"{context}\"\"\"
CONTEXT này có hữu ích để trả lời đúng câu hỏi theo đáp án tham chiếu không?
Trả JSON: {{"useful": true hoặc false, "reason": "ngắn gọn"}}"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def evaluate_ragas(questions: Sequence[str], answers: Sequence[str],
                   contexts: Sequence[Sequence[str]], ground_truths: Sequence[str],
                   *, client: Any | None = None, implementation: str | None = None,
                   store: ResumableStore | None = None) -> dict:
    """Score every row and report which models actually ran."""
    rows = _validate_inputs(questions, answers, contexts, ground_truths)
    store = store or ResumableStore(config.RAGAS_CACHE_PATH, key_field="key")
    client = client or llm.get_client(role="evaluator")
    cache = _CacheView(store)
    requested = (implementation or config.RAGAS_IMPL or "auto").lower()
    if requested not in ("auto", LIBRARY, BUILTIN):
        raise ValueError(f"RAGAS_IMPL must be auto/library/builtin, got {requested!r}")
    impl = LIBRARY if requested == LIBRARY or (requested == "auto" and _library_importable()) else BUILTIN

    def pending(which: str) -> list[dict]:
        return [row for row in rows if cache.get(row, which) is None]

    starts_missing = len(pending(impl))
    if starts_missing != len(rows):
        print(f"  ✓ Reusing {len(rows) - starts_missing}/{len(rows)} cached metric rows")

    if impl == LIBRARY:
        failures = _run_library(pending(LIBRARY), cache, LIBRARY)
        if failures and requested == "auto":
            # Fill only the gaps with the built-in metrics instead of re-spending
            # the whole batch; both paths implement the same four definitions.
            print(f"  ⚠️  ragas library left {len(failures)} row(s) unresolved "
                  f"({[row['question_id'] for row in failures]}); "
                  "scoring those rows with the built-in implementation")
            _run_builtin(failures, cache, BUILTIN, client)
    if impl == BUILTIN:
        _run_builtin(pending(BUILTIN), cache, BUILTIN, client)

    per_question, unresolved, used = [], [], set()
    for row in rows:
        found, values = None, None
        for candidate in (impl, BUILTIN if impl == LIBRARY else LIBRARY):
            values = cache.get(row, candidate)
            if values is not None:
                found = candidate
                break
        if values is None:
            unresolved.append(row["question_id"])
            continue
        used.add(found)
        per_question.append(SimpleNamespace(**values))
    if unresolved:
        raise ValueError(
            "Evaluation incomplete after retries for question ids "
            f"{unresolved}. Re-run to resume from the cache; no partial report is written.")
    cache.flush()
    return {
        "per_question": per_question,
        "metadata": {
            "evaluator_models": sorted({config.EVALUATOR_MODEL, config.EMBEDDING_MODEL}),
            "metric_implementation": "+".join(sorted(used)) or impl,
            "evaluator_llm": config.EVALUATOR_MODEL,
            "evaluator_embeddings": config.EMBEDDING_MODEL,
            "metrics": list(METRICS),
        },
    }


# ---------------------------------------------------------------------------
# Inputs / cache plumbing
# ---------------------------------------------------------------------------


def _validate_inputs(questions, answers, contexts, ground_truths) -> list[dict]:
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("evaluate_ragas requires four equally long lists; got "
                         f"{[len(questions), len(answers), len(contexts), len(ground_truths)]}")
    rows = []
    for index, (question, answer, context_list, ground_truth) in enumerate(
            zip(questions, answers, contexts, ground_truths)):
        if not isinstance(context_list, (list, tuple)):
            raise ValueError(f"contexts[{index}] must be a list of strings.")
        rows.append({
            "question_id": index,
            "question": str(question),
            "answer": str(answer),
            "contexts": [str(item) for item in context_list],
            "ground_truth": str(ground_truth),
        })
    return rows


def _row_key(row: dict, impl: str) -> str:
    digest = hashlib.sha256()
    for part in (impl, row["question"], row["answer"], "\x1f".join(row["contexts"]), row["ground_truth"]):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


class _CacheView:
    """Thin wrapper around the resumable store; every write is flushed immediately
    so a process that stops on quota keeps all rows already scored."""

    def __init__(self, store: ResumableStore):
        self.store = store
        self._pending: dict[str, dict] = {}

    def get(self, row: dict, impl: str) -> dict | None:
        key = _row_key(row, impl)
        if key in self._pending:
            return self._pending[key]["values"]
        record = self.store.get(key)
        return record.get("values") if record else None

    def put(self, row: dict, impl: str, values: dict) -> None:
        key = _row_key(row, impl)
        record = {"key": key, "implementation": impl, "question_id": row["question_id"],
                  "question": row["question"], "values": values}
        self._pending[key] = record
        self.store.upsert(record, save=False)
        self.flush()

    def flush(self) -> None:
        self.store.save()


def _coerce_metric(value: Any, name: str, question_id: int) -> float:
    """Finite [0,1] float or a hard error — NaN is never replaced by a guess."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"question {question_id}: {name} is not numeric ({value!r}).")
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"question {question_id}: {name} is not finite ({value!r}).")
    return min(1.0, max(0.0, number))


# ---------------------------------------------------------------------------
# Implementation 1: the ragas library
# ---------------------------------------------------------------------------


def _library_importable() -> bool:
    try:
        import ragas  # noqa: F401
        from ragas import evaluate as _evaluate  # noqa: F401
        from ragas.metrics import faithfulness, answer_relevancy  # noqa: F401
        from langchain_openai import ChatOpenAI  # noqa: F401
        from datasets import Dataset  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️  ragas library unavailable ({type(exc).__name__}: {exc})")
        return False
    return True


def _run_library(rows: list[dict], cache: _CacheView, impl: str) -> list[dict]:
    """Evaluate in batches, bisecting a failing batch down to single questions.

    Returns the rows that could not be scored, so the caller can decide whether to
    fill only those gaps or to stop.
    """
    if not rows:
        return []
    failures: list[dict] = []
    queue: list[list[dict]] = [rows[i:i + _BATCH] for i in range(0, len(rows), _BATCH)]
    while queue:
        group = queue.pop(0)
        if not group:
            continue
        try:
            values_list = _library_batch(group)
        except Exception as exc:  # noqa: BLE001
            if len(group) == 1:
                print(f"  ⚠️  ragas could not score question id {group[0]['question_id']}: "
                      f"{type(exc).__name__}: {exc}")
                failures.append(group[0])
                continue
            middle = len(group) // 2
            queue[0:0] = [group[:middle], group[middle:]]
            continue
        for row, values in zip(group, values_list):
            cache.put(row, impl, values)
        ids = [row["question_id"] for row in group]
        print(f"    · ragas scored question ids {ids[0]}..{ids[-1]} ({len(ids)} rows)")
    return failures


def _library_batch(rows: list[dict]) -> list[dict]:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import RunConfig, evaluate as ragas_evaluate
    from ragas.metrics import (answer_relevancy, context_precision, context_recall,
                               faithfulness)

    from src.embeddings import LocalEmbeddings

    evaluator = ChatOpenAI(
        model=config.EVALUATOR_MODEL,
        base_url=llm.resolve_base_url(),
        api_key=llm.require_api_key(),
        temperature=0.0,
        timeout=config.LLM_TIMEOUT_S,
        max_retries=config.LLM_MAX_RETRIES,
    )
    try:
        answer_relevancy.strictness = max(1, config.RAGAS_STRICTNESS)
    except AttributeError:
        pass

    dataset = Dataset.from_dict({
        "question": [row["question"] for row in rows],
        "answer": [row["answer"] for row in rows],
        "contexts": [row["contexts"] for row in rows],
        "ground_truth": [row["ground_truth"] for row in rows],
    })
    run_config = RunConfig(timeout=config.LLM_TIMEOUT_S,
                           max_retries=config.LLM_MAX_RETRIES,
                           max_workers=max(1, config.RAGAS_MAX_WORKERS))
    # Pass only what the INSTALLED ragas accepts. ragas 0.1.22's evaluate() has no
    # `show_progress` (it was removed after the 0.1.10 lower bound in
    # requirements.txt), and passing it raised TypeError for every row before this
    # check — which is exactly why the pre-flight runs before the full batch.
    import inspect
    supported = set(inspect.signature(ragas_evaluate).parameters)
    requested = {
        "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
        "llm": evaluator,
        "embeddings": LocalEmbeddings(),
        "run_config": run_config,
        "raise_exceptions": False,
    }
    unsupported = sorted(set(requested) - supported)
    if unsupported:
        print(f"  ⚠️  ragas {getattr(__import__('ragas'), '__version__', '?')} does not accept "
              f"{unsupported}; omitting")
    result = ragas_evaluate(dataset, **{k: v for k, v in requested.items() if k in supported})
    frame = result.to_pandas()
    if len(frame) != len(rows):
        raise ValueError(f"ragas returned {len(frame)} rows for {len(rows)} inputs.")
    out: list[dict] = []
    for position, row in enumerate(rows):
        record = frame.iloc[position]
        out.append({name: _coerce_metric(record[name], name, row["question_id"])
                    for name in METRICS})
    return out


# ---------------------------------------------------------------------------
# Implementation 2: the same metrics on the shared client
# ---------------------------------------------------------------------------


def _run_builtin(rows: list[dict], cache: _CacheView, impl: str, client: Any) -> None:
    for row in rows:
        attempt_values = None
        last_error: Exception | None = None
        for _ in range(2):
            try:
                attempt_values = _builtin_one(row, client)
                break
            except Exception as exc:  # noqa: BLE001 - retried once, then reported
                last_error = exc
        if attempt_values is None:
            raise ValueError(
                f"built-in metrics failed for question id {row['question_id']}: "
                f"{type(last_error).__name__}: {last_error}") from last_error
        cache.put(row, impl, attempt_values)
        print(f"    · builtin row {row['question_id']} scored")


def _json_call(client: Any, *, prompt: str, label: str) -> dict:
    return llm.chat_json(client, model=config.EVALUATOR_MODEL,
                         messages=[{"role": "system", "content": _JSON_ONLY},
                                   {"role": "user", "content": prompt}],
                         label=label)


def _claim_statements(client: Any, text: str, label: str) -> list[str]:
    parsed = _json_call(client, prompt=CLAIM_PROMPT.format(text=text), label=label)
    values = parsed.get("statements")
    if not isinstance(values, list):
        raise ValueError(f"{label}: 'statements' must be a list.")
    return [str(item).strip() for item in values if str(item).strip()]


def _supported_fraction(client: Any, statements: Iterable[str], contexts: list[str],
                        label: str) -> float:
    items = list(statements)
    if not items:
        return 1.0
    evidence = "\n\n".join(contexts) if contexts else "(không có context)"
    supported = 0
    for position, statement in enumerate(items):
        verdict = _json_call(client, prompt=SUPPORT_PROMPT.format(context=evidence, statement=statement),
                            label=f"{label} support #{position}")
        if str(verdict.get("verdict", "")).strip().lower().startswith("y"):
            supported += 1
    return supported / len(items)


def _answer_relevancy(client: Any, row: dict) -> float:
    import numpy as np
    from src.embeddings import encode
    parsed = _json_call(client, prompt=RELEVANCY_PROMPT.format(answer=row["answer"]),
                        label=f"relevancy q{row['question_id']}")
    generated = str(parsed.get("question") or "").strip()
    if not generated:
        raise ValueError(f"question {row['question_id']}: generated question is empty.")
    original_vector, generated_vector = encode([row["question"], generated])
    similarity = float(np.dot(original_vector, generated_vector))
    return min(1.0, max(0.0, similarity))


def _context_precision(client: Any, row: dict) -> float:
    """ragas definition: sum of precision@k over relevant contexts / relevant count."""
    hits = 0
    accumulated = 0.0
    for position, context in enumerate(row["contexts"]):
        parsed = _json_call(
            client,
            prompt=PRECISION_PROMPT.format(question=row["question"], ground_truth=row["ground_truth"],
                                          context=context),
            label=f"precision q{row['question_id']} c{position}")
        if bool(parsed.get("useful")):
            hits += 1
            accumulated += hits / (position + 1)
    return accumulated / hits if hits else 0.0


def _builtin_one(row: dict, client: Any) -> dict:
    label = f"q{row['question_id']}"
    answer_statements = _claim_statements(client, row["answer"], f"{label} claims(answer)")
    reference_statements = _claim_statements(client, row["ground_truth"], f"{label} claims(reference)")
    return {
        "faithfulness": _supported_fraction(client, answer_statements, row["contexts"], f"{label} faithful"),
        "answer_relevancy": _answer_relevancy(client, row),
        "context_precision": _context_precision(client, row),
        "context_recall": _supported_fraction(client, reference_statements, row["contexts"],
                                               f"{label} recall"),
    }
