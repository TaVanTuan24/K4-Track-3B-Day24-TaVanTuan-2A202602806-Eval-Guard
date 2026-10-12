"""Provided runners. Live experiments run only via --live; no canned submission results."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path

from config import (ADVERSARIAL_SET_PATH, ANSWERS_PATH, ARTIFACTS_DIR, CLEAN_SET_PATH,
                    HUMAN_LABELS_PATH, OUTPUT_SET_PATH, PAIR_QUESTION_IDS, TEST_SET_PATH)
import src.llm_client as llm
from src.experiment_support import (JUDGE_MODEL, generation_models, guard_models, judge_binary,
                                    metadata, model_ids, read_json, write_json)
from src.report_contracts import PAIR_SOURCES, validate_answers, validate_phase_a, validate_phase_b, validate_phase_c


def run_phase_a():
    from src import phase_a_ragas as phase
    answers = read_json(ANSWERS_PATH)
    validate_answers(answers, read_json(TEST_SET_PATH))
    evaluation = phase.run_ragas_50q(answers)
    if not isinstance(evaluation, phase.RagasEvaluation):
        raise ValueError("Task 2 must return RagasEvaluation with actual evaluator_models from m4.")
    evaluator_models = model_ids(evaluation.evaluator_models, "evaluator models")
    if not evaluator_models:
        raise ValueError("m4 must report actual evaluator models; no JUDGE_MODEL fallback.")
    results = evaluation.results
    bottom = phase.bottom_10(results)
    clusters = phase.cluster_analysis(results)
    path = Path(ARTIFACTS_DIR) / "ragas_50q.json"
    report = phase.build_phase_a_report(results, clusters, bottom)
    report["metadata"] = metadata([ANSWERS_PATH, TEST_SET_PATH], models={
        "generation": generation_models(answers), "evaluator": evaluator_models,
        "judge": [], "guard": []})
    validate_phase_a(report, answers)
    write_json(path, report)
    return report


def _input_hash(value):
    import hashlib
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def run_phase_b():
    from src import cache_keys
    from src import phase_b_judge as phase
    from src.checkpoint import ResumableStore
    answers = read_json(ANSWERS_PATH)
    questions = read_json(TEST_SET_PATH)
    humans = read_json(HUMAN_LABELS_PATH)
    validate_answers(answers, questions)
    by_answer = {row["id"]: row for row in answers}
    by_question = {row["id"]: row for row in questions}
    by_human = {row["question_id"]: row for row in humans}

    # Resumable caches: a run interrupted by quota keeps the judge calls it already
    # paid for. Each entry's fingerprint covers the judge model, the prompt version and
    # the full inputs (question, both candidates / supplied answer, reference), so
    # changing the judge model or the prompt invalidates the row instead of silently
    # reusing a verdict produced by a different configuration.
    cache_dir = Path(ARTIFACTS_DIR) / "cache"
    pair_store = ResumableStore(cache_dir / "phase_b_pairwise.json", key_field="question_id")
    binary_store = ResumableStore(cache_dir / "phase_b_binary.json", key_field="question_id")
    judge_model = phase.JUDGE_MODEL

    results = []
    for qid in PAIR_QUESTION_IDS:
        answer_a = by_answer[qid]["answer"]
        answer_b = by_human[qid]["model_answer"]
        question = by_question[qid]["question"]
        reference = by_question[qid]["ground_truth"]
        key = cache_keys.pairwise_judge_key(
            judge_model=judge_model, prompt_version=cache_keys.PAIRWISE_PROMPT_VERSION,
            question=question, answer_a=answer_a, answer_b=answer_b, reference=reference)
        cached = pair_store.get(qid)
        if cached and cached.get("fingerprint") == key:
            results.append(cached["result"])
            continue
        result = phase.swap_and_average(question, answer_a, answer_b, reference=reference)
        row = {"question_id": qid, **asdict(result)}
        pair_store.upsert({"question_id": qid, "fingerprint": key,
                           "judge_model": judge_model,
                           "prompt_version": cache_keys.PAIRWISE_PROMPT_VERSION,
                           "answer_a_hash": _input_hash(answer_a),
                           "answer_b_hash": _input_hash(answer_b),
                           "reference_hash": _input_hash(reference),
                           "result": row})
        results.append(row)

    judgments = []
    binary_client = llm.get_client(role="judge")
    for human in humans:
        qid = human["question_id"]
        reference = by_question[qid]["ground_truth"]
        key = cache_keys.binary_judge_key(
            judge_model=judge_model, prompt_version=cache_keys.BINARY_PROMPT_VERSION,
            question=human["question"], model_answer=human["model_answer"], reference=reference)
        cached = binary_store.get(qid)
        if cached and cached.get("fingerprint") == key:
            judgments.append(cached["result"])
            continue
        # Exact supplied answer. Never pass human_label or human_note into judge_binary.
        result = judge_binary(human["question"], human["model_answer"],
                              reference, client=binary_client)
        row = {"question_id": qid, "question": human["question"],
               "model_answer": human["model_answer"], **result}
        binary_store.upsert({"question_id": qid, "fingerprint": key,
                             "judge_model": judge_model,
                             "prompt_version": cache_keys.BINARY_PROMPT_VERSION,
                             "model_answer_hash": _input_hash(human["model_answer"]),
                             "reference_hash": _input_hash(reference),
                             "result": row})
        judgments.append(row)
    judge_labels = [row["label"] for row in judgments]
    human_labels = [row["human_label"] for row in humans]
    report = {
        "metadata": metadata([ANSWERS_PATH, TEST_SET_PATH, HUMAN_LABELS_PATH], models={
            "generation": generation_models(answers), "evaluator": [],
            "judge": [phase.JUDGE_MODEL, JUDGE_MODEL], "guard": []}),
        "pair_sources": dict(PAIR_SOURCES),
        "pairwise_results": results, "binary_judgments": judgments,
        "calibration": {"question_ids": [row["question_id"] for row in judgments],
                        "judge_labels": judge_labels, "human_labels": human_labels,
                        "cohen_kappa": phase.cohen_kappa(judge_labels, human_labels)},
        "bias_report": phase.bias_report([
            phase.JudgeResult(**{k: v for k, v in row.items() if k != "question_id"})
            for row in results]),
    }
    validate_phase_b(report, answers, questions, humans)
    write_json(Path(ARTIFACTS_DIR) / "judge_results.json", report)
    return report


def _summarize_input(rows):
    passed = sum(row["passed"] for row in rows)
    return {"total": len(rows), "passed": passed, "pass_rate": passed / len(rows), "results": rows}


def run_phase_c(n_samples=20):
    from src import phase_c_guard as phase
    if n_samples < 20:
        raise ValueError("Submission latency needs at least 20 timed samples.")
    attacks, clean, outputs = [read_json(path) for path in
                              (ADVERSARIAL_SET_PATH, CLEAN_SET_PATH, OUTPUT_SET_PATH)]
    analyzer, anonymizer = phase.setup_presidio()
    # One combined input run, not four replays of the attack suite.
    # Each stage owns its NeMo instance, avoiding HTTP clients reused across closed loops.
    input_rails = phase.setup_nemo_rails()
    used_guard_models = guard_models(input_rails)
    input_rows = phase.run_adversarial_suite(
        attacks + clean, rails=input_rails, analyzer=analyzer, anonymizer=anonymizer)
    attack_ids = {row["id"] for row in attacks}
    clean_ids = {row["id"] for row in clean}
    attack_rows = [row for row in input_rows if row["id"] in attack_ids]
    clean_rows = [row for row in input_rows if row["id"] in clean_ids]
    if len(attack_rows) != 20 or len(clean_rows) != len(clean):
        raise ValueError("Input suite did not return the full attack and clean sets.")

    async def check_outputs():
        rails = phase.setup_nemo_rails()
        used_guard_models.extend(guard_models(rails))
        rows = []
        for case in outputs:
            result = await phase.check_output_rail(case["question"], case["answer"], rails=rails)
            rows.append({**case, **result,
                         "passed": result["safe"] == (case["expected"] == "safe")})
        return rows
    output_rows = asyncio.run(check_outputs())
    latency_rails = phase.setup_nemo_rails()
    used_guard_models.extend(guard_models(latency_rails))
    latency = phase.measure_p95_latency(
        [row["input"] for row in clean], n_runs=n_samples,
        rails=latency_rails, analyzer=analyzer, anonymizer=anonymizer)
    clean_summary = _summarize_input(clean_rows)
    clean_summary["false_positive_rate"] = sum(
        row["actual"] == "blocked" for row in clean_rows) / len(clean_rows)
    report = {
        "metadata": metadata([ADVERSARIAL_SET_PATH, CLEAN_SET_PATH, OUTPUT_SET_PATH], models={
            "generation": [], "evaluator": [], "judge": [], "guard": used_guard_models}),
        "adversarial": _summarize_input(attack_rows), "clean": clean_summary,
        "output_results": output_rows, "latency": latency,
    }
    # Save genuine failing experiments too; checker enforces 15/20 and clean controls.
    validate_phase_c(report, attacks, clean, outputs, require_quality=False)
    write_json(Path(ARTIFACTS_DIR) / "guard_results.json", report)
    return report
