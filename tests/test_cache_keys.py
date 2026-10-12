"""Cache-key tests: a changed evaluator/judge configuration must invalidate the cache.

The live runs are resumable, so a wrong cache key is worse than a cache miss: it would
silently serve scores or verdicts produced by a *different model or prompt* as if they
belonged to the current configuration. These tests pin both directions:

* changing a model / prompt version / input ⇒ the key changes (cache miss, re-compute);
* leaving everything unchanged ⇒ the key is stable (cache hit, no API spend).

Nothing here calls an API.
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest

import config
from src import cache_keys

pytestmark = pytest.mark.support

ROW = {
    "question_id": 1,
    "question": "Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?",
    "answer": "3 ngày làm việc có lương.",
    "contexts": ["[nghỉ phép đặc biệt | quy định] Kết hôn: 3 ngày làm việc"],
    "ground_truth": "Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn.",
}


def _ragas_key(**overrides):
    kwargs = dict(implementation="library", evaluator_model="eval-a", embedding_model="emb-a",
                  metric_version="ragas-0.1.22", question=ROW["question"], answer=ROW["answer"],
                  contexts=ROW["contexts"], ground_truth=ROW["ground_truth"])
    kwargs.update(overrides)
    return cache_keys.ragas_row_key(**kwargs)


# ---------------------------------------------------------------------------
# Phase A
# ---------------------------------------------------------------------------


def test_ragas_key_is_stable_for_identical_inputs():
    assert _ragas_key() == _ragas_key()


@pytest.mark.parametrize("field,value", [
    ("evaluator_model", "eval-b"),
    ("embedding_model", "emb-b"),
    ("implementation", "builtin"),
    ("metric_version", "builtin-metrics-v2"),
    ("question", "câu hỏi khác"),
    ("answer", "câu trả lời khác"),
    ("contexts", ["context khác"]),
    ("ground_truth", "reference khác"),
])
def test_ragas_key_changes_when_configuration_or_input_changes(field, value):
    assert _ragas_key(**{field: value}) != _ragas_key(), f"{field} must invalidate the key"


def test_ragas_row_key_reads_the_live_evaluator_configuration(monkeypatch):
    """The m4 cache key follows config, so switching EVALUATOR_MODEL invalidates."""
    from src import m4_eval

    baseline = m4_eval._row_key(ROW, m4_eval.LIBRARY)
    monkeypatch.setattr(config, "EVALUATOR_MODEL", "another-evaluator")
    assert m4_eval._row_key(ROW, m4_eval.LIBRARY) != baseline

    monkeypatch.setattr(config, "EVALUATOR_MODEL", "eval-a")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "another-embedding")
    assert m4_eval._row_key(ROW, m4_eval.LIBRARY) != baseline


def test_ragas_row_key_separates_implementations(monkeypatch):
    from src import m4_eval

    assert m4_eval._row_key(ROW, m4_eval.LIBRARY) != m4_eval._row_key(ROW, m4_eval.BUILTIN)


def test_metric_version_tracks_the_installed_ragas(monkeypatch):
    from src import m4_eval

    version = m4_eval._metric_version(m4_eval.LIBRARY)
    assert version.startswith("ragas-")
    assert m4_eval._metric_version(m4_eval.BUILTIN) == cache_keys.BUILTIN_METRIC_VERSION


# ---------------------------------------------------------------------------
# Phase B
# ---------------------------------------------------------------------------


def _pairwise_key(**overrides):
    kwargs = dict(judge_model="judge-a", prompt_version="pairwise-judge-v1",
                  question="Q", answer_a="A", answer_b="B", reference="R")
    kwargs.update(overrides)
    return cache_keys.pairwise_judge_key(**kwargs)


def _binary_key(**overrides):
    kwargs = dict(judge_model="judge-a", prompt_version="binary-judge-v1",
                  question="Q", model_answer="A", reference="R")
    kwargs.update(overrides)
    return cache_keys.binary_judge_key(**kwargs)


def test_pairwise_key_is_stable_and_configuration_sensitive():
    assert _pairwise_key() == _pairwise_key()
    for field, value in (("judge_model", "judge-b"), ("prompt_version", "pairwise-judge-v2"),
                         ("question", "Q2"), ("answer_a", "A2"), ("answer_b", "B2"),
                         ("reference", "R2")):
        assert _pairwise_key(**{field: value}) != _pairwise_key(), f"{field} must invalidate"


def test_binary_key_is_stable_and_configuration_sensitive():
    assert _binary_key() == _binary_key()
    for field, value in (("judge_model", "judge-b"), ("prompt_version", "binary-judge-v2"),
                         ("question", "Q2"), ("model_answer", "A2"), ("reference", "R2")):
        assert _binary_key(**{field: value}) != _binary_key(), f"{field} must invalidate"


def test_judge_runner_invalidates_cache_on_model_change(tmp_path, monkeypatch, openai_client):
    """End-to-end: second run hits the cache; a new judge model re-runs the calls."""
    from src import experiment_runners as runners
    from src import phase_b_judge as phase

    questions = runners.read_json(config.TEST_SET_PATH)
    humans = runners.read_json(config.HUMAN_LABELS_PATH)
    answers = [{**q, "answer": "PIPELINE " + str(q["id"]), "contexts": ["c"],
                "generation_mode": "llm", "generation_model": "generation-test"}
               for q in questions]
    paths = {}
    for name, rows in (("test_set_50q.json", questions), ("human_labels_10q.json", humans),
                       ("answers_50q.json", answers)):
        paths[name] = tmp_path / name
        runners.write_json(paths[name], rows)
    monkeypatch.setattr(runners, "TEST_SET_PATH", paths["test_set_50q.json"])
    monkeypatch.setattr(runners, "HUMAN_LABELS_PATH", paths["human_labels_10q.json"])
    monkeypatch.setattr(runners, "ANSWERS_PATH", paths["answers_50q.json"])
    monkeypatch.setattr(runners, "ARTIFACTS_DIR", tmp_path / "artifacts")

    def swap(question, answer_a, answer_b, reference=""):
        assert reference
        return phase.JudgeResult(question, answer_a, answer_b, "A", "A", "A", "r1", "r2",
                                 True, {"A": .9, "B": .2}, {"A": .9, "B": .2})

    swap_calls = Mock(side_effect=swap)
    monkeypatch.setattr(phase, "swap_and_average", swap_calls)

    labels = {h["model_answer"]: h["human_label"] for h in humans}
    binary_calls = Mock(side_effect=lambda q, a, r, client=None:
                        {"label": labels[a], "reasoning": "mock reference evaluation"})
    monkeypatch.setattr(runners, "judge_binary", binary_calls)
    monkeypatch.setattr(phase, "cohen_kappa", lambda a, b: 1)

    def bias(results):
        decisive = sum(1 for r in results if r.final_winner != "tie")
        a_longer = sum(1 for r in results
                       if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b))
        b_longer = sum(1 for r in results
                       if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a))
        longer = a_longer + b_longer
        return {"total_judged": len(results),
                "position_bias_count": sum(1 for r in results if not r.position_consistent),
                "position_bias_rate": (sum(1 for r in results if not r.position_consistent)
                                       / len(results)),
                "verbosity_bias": (longer / decisive) if decisive else 0.0,
                "interpretation": "mock interpretation",
                "verbosity_details": {"a_wins_a_longer": a_longer, "b_wins_b_longer": b_longer,
                                      "total_decisive": decisive}}
    monkeypatch.setattr(phase, "bias_report", bias)

    # 1) cold run pays for every call
    runners.run_phase_b()
    assert swap_calls.call_count == 5 and binary_calls.call_count == 10

    # 2) warm run reuses the cache: no API calls at all
    swap_calls.reset_mock()
    binary_calls.reset_mock()
    runners.run_phase_b()
    assert swap_calls.call_count == 0, "unchanged inputs must hit the cache"
    assert binary_calls.call_count == 0, "unchanged inputs must hit the cache"

    # 3) a different judge model invalidates both caches
    monkeypatch.setattr(phase, "JUDGE_MODEL", "judge-other-model")
    runners.run_phase_b()
    assert swap_calls.call_count == 5 and binary_calls.call_count == 10, \
        "changing the judge model must re-run the judge calls"


# ---------------------------------------------------------------------------
# Secrets must never become part of a fingerprint
# ---------------------------------------------------------------------------


def test_cache_keys_ignore_the_api_key(monkeypatch):
    """Rotating a credential must not discard completed work, and must not leak into
    a checkpoint file on disk."""
    monkeypatch.setattr(config, "API_KEY", "sk-super-secret-value")
    with_first = (_ragas_key(), _pairwise_key(), _binary_key())
    monkeypatch.setattr(config, "API_KEY", "sk-a-completely-different-secret")
    with_second = (_ragas_key(), _pairwise_key(), _binary_key())
    assert with_first == with_second

    for key in with_first:
        assert "secret" not in key.lower()
    # The module exposes a guard the tests use to assert material never carries a key.
    cache_keys.assert_no_secret("sk-super-secret-value",
                               [ROW["question"], ROW["answer"], *ROW["contexts"]])
