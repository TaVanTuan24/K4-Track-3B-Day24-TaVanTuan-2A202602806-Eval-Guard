"""Re-export ONLY the Task 3 diagnosis text in `artifacts/ragas_50q.json`.

Why this exists
---------------
`bottom_10[].diagnosis` and `suggested_fix` are derived annotations produced by Task 3,
not measurements. The original annotations came from a pure metric→cause table, which
labels an answer of "Không tìm thấy." as "LLM hallucinating" — the opposite of what
happened, since such an answer contains no unsupported claim. Task 3 now refines the
label with evidence present in the row (answer text, retrieved contexts, reference).

Re-running the Phase A runner to pick that up would rewrite `metadata.created_at`,
falsely claiming the RAGAS evaluation happened now. This tool therefore rebuilds only
the annotation field from the metrics ALREADY in the artifact:

* No API call, no cache, no re-evaluation.
* `rank`, `question_id`, `distribution`, `question`, `avg_score`, `worst_metric` must be
  byte-identical to what is stored; the tool aborts otherwise, so ranking can never be
  rewritten to make an explanation fit.
* `per_question`, `per_distribution`, `failure_clusters` and `metadata` are not touched,
  and are re-checked before writing.
* Idempotent: a second run is a no-op.

    python scripts/refresh_phase_a_annotations.py --check   # verify, change nothing
    python scripts/refresh_phase_a_annotations.py           # re-export annotations
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.checkpoint import atomic_write_json
from src.phase_a_ragas import RagasResult, bottom_10

ARTIFACT = Path(config.ARTIFACTS_DIR) / "ragas_50q.json"
IDENTITY_FIELDS = ("rank", "question_id", "distribution", "question", "avg_score", "worst_metric")
UNTOUCHED_KEYS = ("total_questions", "per_question", "per_distribution", "failure_clusters", "metadata")


def rebuild(payload: dict) -> list[dict]:
    """Recompute bottom_10 from the artifact's own measured metrics."""
    results = [
        RagasResult(row["question_id"], row["distribution"], row["question"], row["answer"],
                    row["contexts"], row["ground_truth"], row["faithfulness"],
                    row["answer_relevancy"], row["context_precision"], row["context_recall"])
        for row in payload["per_question"]
    ]
    return bottom_10(results)


def verify_identity(stored: list[dict], rebuilt: list[dict]) -> None:
    if len(stored) != len(rebuilt):
        raise SystemExit(f"bottom_10 length changed: {len(stored)} -> {len(rebuilt)}; refusing.")
    for old, new in zip(stored, rebuilt):
        for field in IDENTITY_FIELDS:
            if old.get(field) != new.get(field):
                raise SystemExit(
                    f"Refusing to write: bottom_10[{old.get('rank')}].{field} would change "
                    f"({old.get(field)!r} -> {new.get(field)!r}). Annotations must never "
                    "rewrite ranking or metrics.")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description="Re-export Phase A diagnosis annotations (offline)")
    parser.add_argument("--check", action="store_true", help="Verify only; exit 1 when stale")
    parser.add_argument("--root", type=Path, default=Path(config.ARTIFACTS_DIR))
    args = parser.parse_args(argv)

    path = args.root / ARTIFACT.name
    if not path.is_file():
        raise SystemExit(f"Artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    before = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    stored = payload.get("bottom_10")
    if not isinstance(stored, list) or not stored:
        raise SystemExit(f"{path.name}: no bottom_10 to refresh.")

    rebuilt = rebuild(payload)
    verify_identity(stored, rebuilt)

    changes = [(old["question_id"], old.get("diagnosis"), new["diagnosis"])
               for old, new in zip(stored, rebuilt) if old.get("diagnosis") != new["diagnosis"]]
    if not changes:
        print(f"OK    {path.name}: diagnosis/fix already match Task 3 ({len(stored)} rows)")
        return 0

    if args.check:
        print(f"STALE {path.name}: {len(changes)} of {len(stored)} diagnosis rows differ")
        for qid, old, new in changes:
            print(f"    id {qid}: {str(old)[:58]}")
            print(f"        -> {str(new)[:58]}")
        return 1

    guards = {key: json.dumps(payload.get(key), ensure_ascii=False, sort_keys=True)
              for key in UNTOUCHED_KEYS}
    payload["bottom_10"] = rebuilt
    after = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for key in UNTOUCHED_KEYS:
        now = json.dumps(payload.get(key), ensure_ascii=False, sort_keys=True)
        if now != guards[key]:
            raise SystemExit(f"{path.name}: {key} changed unexpectedly; refusing to write.")
    if before == after:
        raise SystemExit(f"{path.name}: differences were reported but nothing changed; refusing.")

    atomic_write_json(path, payload)
    print(f"UPDATE {path.name}: re-exported {len(changes)} of {len(stored)} diagnosis rows")
    for qid, old, new in changes:
        print(f"    id {qid}: {str(old)[:58]}")
        print(f"        -> {str(new)[:58]}")
    print("    rank, metrics, per_question, per_distribution, failure_clusters, metadata untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
