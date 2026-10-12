"""Pre-flight for Phase A: score a few rows with m4 BEFORE spending the full batch.

    python scripts/check_evaluator.py --live            # first 3 answers, ragas library
    python scripts/check_evaluator.py --live --rows 2 --impl builtin
    python scripts/check_evaluator.py --live --impl auto

RAGAS drives its four metrics through many small LLM calls, and a tool that cannot
talk to the configured endpoint fails *after* hundreds of paid calls if it is only
tested on the whole set. This prints, per row, the four scores, which implementation
produced them, and the wall-clock time, so the full 50-row run is only started on a
path that is known to work. Cached rows are reused, so this is cheap to repeat.

`--impl auto` (the default) uses the ragas library and falls back to the built-in
implementation of the same four metrics if the library cannot call the endpoint.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import src.llm_client as llm
from src.checkpoint import ResumableStore
from src.m4_eval import METRICS, evaluate_ragas
from src.phase_a_ragas import load_answers


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description="Phase A evaluator pre-flight (real API calls)")
    parser.add_argument("--live", action="store_true", help="Acknowledge real API calls")
    parser.add_argument("--rows", type=int, default=3, help="How many answers to score")
    parser.add_argument("--impl", choices=("auto", "library", "builtin"), default="auto")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("This pre-flight makes real API calls. Pass --live.")
    try:
        llm.require_api_key()
    except llm.MissingCredential as error:
        parser.error(str(error))

    answers = load_answers()[:max(1, args.rows)]
    print("=" * 66)
    print(f"Phase A pre-flight — {len(answers)} row(s), implementation={args.impl}")
    print(f"endpoint={llm.resolve_base_url()} evaluator={config.EVALUATOR_MODEL} "
          f"embeddings={config.EMBEDDING_MODEL}")
    print("=" * 66)
    llm.set_pacing(config.LLM_MIN_INTERVAL_S)

    # A dedicated cache file keeps this pre-flight from mixing rows into the real
    # Phase A cache under a different implementation.
    store = ResumableStore(os.path.join(config.CACHE_DIR, f"ragas_preflight_{args.impl}.json"),
                           key_field="key")
    started = time.time()
    try:
        raw = evaluate_ragas(
            [row["question"] for row in answers],
            [row["answer"] for row in answers],
            [row["contexts"] for row in answers],
            [row["ground_truth"] for row in answers],
            implementation=args.impl, store=store)
    except Exception as error:  # noqa: BLE001 - the point of the pre-flight
        print(f"FAIL  evaluator — {type(error).__name__}: {error}")
        print(f"      elapsed {time.time() - started:.1f}s")
        return 1

    metadata = raw["metadata"]
    for row, scores in zip(answers, raw["per_question"]):
        values = {name: getattr(scores, name) for name in METRICS}
        print(f"  id={row['id']:<3} " + " ".join(f"{k}={v:.3f}" for k, v in values.items()))
    elapsed = time.time() - started
    print("-" * 66)
    print(f"PASS  evaluator — rows={len(raw['per_question'])} "
          f"implementation={metadata['metric_implementation']} "
          f"evaluator_models={metadata['evaluator_models']}")
    print(f"      elapsed {elapsed:.1f}s → {elapsed / max(1, len(answers)):.1f}s/row; "
          f"projected for 50 rows ≈ {elapsed / max(1, len(answers)) * 50 / 60:.0f} min "
          f"(cached rows cost nothing)")
    print("Safe to run: python scripts/run_experiments.py --phase a --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
