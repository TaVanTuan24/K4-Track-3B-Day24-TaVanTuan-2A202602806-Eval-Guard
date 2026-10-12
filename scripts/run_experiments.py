"""Run submitted experiments explicitly. Default tests never call paid APIs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import src.llm_client as llm
from src.experiment_support import require_completed
from src.experiment_runners import run_phase_a, run_phase_b, run_phase_c


def main(argv=None):
    # Vietnamese report text must not crash when stdout is a pipe (cp1252 default),
    # and must be flushed line by line so a long live run is observable.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(
        description="RAG eval/guard experiments (real API calls against the configured "
                    "OpenAI-compatible endpoint — Gemini free tier by default)")
    parser.add_argument("--phase", choices=("a", "b", "c", "all"), required=True)
    parser.add_argument("--live", action="store_true", help="Acknowledge API use and network dependency")
    parser.add_argument("--latency-samples", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Real experiments require --live. Use python -m pytest for offline unit tests.")
    try:
        llm.require_api_key()
    except llm.MissingCredential as error:
        parser.error(str(error))

    print(f"Endpoint : {llm.resolve_base_url()}")
    print(f"Models   : generation={config.GENERATION_MODEL} judge={config.JUDGE_MODEL} "
          f"evaluator={config.EVALUATOR_MODEL}")
    print(f"Pacing   : {config.LLM_MIN_INTERVAL_S}s between calls, "
          f"concurrency {config.LLM_MAX_CONCURRENCY}, "
          f"{config.LLM_MAX_RETRIES} retries on 429/5xx")
    # Free-tier pacing: space requests out instead of hammering the quota.
    llm.set_pacing(config.LLM_MIN_INTERVAL_S)

    from src import phase_a_ragas, phase_b_judge, phase_c_guard
    modules = {"a": phase_a_ragas, "b": phase_b_judge, "c": phase_c_guard}
    selected = list(modules) if args.phase == "all" else [args.phase]
    for name in selected:
        require_completed(modules[name])
    runners = {"a": run_phase_a, "b": run_phase_b, "c": lambda: run_phase_c(args.latency_samples)}
    for name in selected:
        runners[name]()
        print(f"Phase {name.upper()} saved to artifacts/.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        raise SystemExit(f"Experiment failed: {error}")
