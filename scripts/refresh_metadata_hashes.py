"""Recompute `metadata.source_sha256` for the committed artifacts — metadata only.

Why this tool exists
--------------------
The artifacts record the SHA-256 of their input files, and `src/report_contracts.py`
recomputes those hashes with `Path.read_bytes()` to detect stale evidence. Those bytes
depend on the checkout's line endings, so a repository-wide change of the input bytes —
here, adding `.gitattributes` and normalizing the working tree from CRLF to LF so the
hashes are identical on Windows, Linux and CI — legitimately invalidates every recorded
hash at once.

This tool is a deliberately narrow, auditable way to re-record them:

* It updates **only** `metadata.source_sha256`. Metrics, per-question rows, bottom-10
  ordering, cluster matrix, bias/calibration numbers, latency samples, `created_at`,
  `run_kind`, `format_version` and the model roles are left untouched.
* It never calls an API. Re-running RAGAS / Judge / Guardrails to fix a hash would be
  waste; a hash is not a measurement.
* It always computes the hashes from the real files. No checksum is hardcoded, and the
  check is never skipped — `--check` fails loudly instead of "repairing" anything.
* It refuses to run when a source file is missing, so a typo cannot silently drop a
  provenance entry.

Typical use:

    python scripts/refresh_metadata_hashes.py --check          # verify, change nothing
    python scripts/refresh_metadata_hashes.py --reason "..."   # re-record, prints a diff
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.checkpoint import atomic_write_json

# Every file name that may appear in metadata.source_sha256 -> its real path.
SOURCE_PATHS = {
    Path(config.ANSWERS_PATH).name: Path(config.ANSWERS_PATH),
    Path(config.TEST_SET_PATH).name: Path(config.TEST_SET_PATH),
    Path(config.HUMAN_LABELS_PATH).name: Path(config.HUMAN_LABELS_PATH),
    Path(config.ADVERSARIAL_SET_PATH).name: Path(config.ADVERSARIAL_SET_PATH),
    Path(config.CLEAN_SET_PATH).name: Path(config.CLEAN_SET_PATH),
    Path(config.OUTPUT_SET_PATH).name: Path(config.OUTPUT_SET_PATH),
}

ARTIFACTS = ("ragas_50q.json", "judge_results.json", "guard_results.json")

# Fields that must never be touched by this tool. If any of them differ before/after,
# the tool aborts instead of writing.
IMMUTABLE_KEYS = ("format_version", "run_kind", "models", "created_at")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(recorded: dict) -> dict:
    """Fresh hashes for exactly the keys already recorded (never a different set)."""
    updated = {}
    for name in recorded:
        path = SOURCE_PATHS.get(name)
        if path is None:
            raise SystemExit(f"Unknown source file in metadata: {name!r}. "
                             f"Known: {sorted(SOURCE_PATHS)}")
        if not path.is_file():
            raise SystemExit(f"Source file is missing: {path}")
        updated[name] = sha256_of(path)
    return updated


def process(path: Path, *, check: bool, reason: str) -> tuple[bool, list[str]]:
    """Return (changed, human-readable notes) for one artifact."""
    if not path.is_file():
        raise SystemExit(f"Artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("source_sha256"), dict):
        raise SystemExit(f"{path.name}: no metadata.source_sha256 to refresh.")
    recorded = metadata["source_sha256"]
    updated = recompute(recorded)

    differences = [f"{name}: {recorded[name][:12]}… -> {updated[name][:12]}…"
                   for name in recorded if recorded[name] != updated[name]]
    if not differences:
        return False, [f"{path.name}: all {len(recorded)} source hashes already match"]

    if check:
        return True, [f"{path.name}: STALE ({len(differences)} of {len(recorded)} hashes differ)",
                      *[f"    {line}" for line in differences]]

    # Guard: nothing outside metadata.source_sha256 may change.
    before = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    metadata["source_sha256"] = updated
    for key in IMMUTABLE_KEYS:
        if key not in metadata:
            raise SystemExit(f"{path.name}: metadata.{key} is missing; refusing to write.")
    after = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if before == after:
        raise SystemExit(f"{path.name}: recomputation produced no change but differences "
                         "were reported; refusing to write.")

    atomic_write_json(path, payload)
    return True, [
        f"{path.name}: re-recorded {len(differences)} hash(es) — reason: {reason}",
        *[f"    {line}" for line in differences],
        "    metrics, rankings, timestamps and model roles untouched",
    ]


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    parser = argparse.ArgumentParser(description="Recompute artifact metadata.source_sha256 (no API calls)")
    parser.add_argument("--check", action="store_true",
                        help="Only verify; exit 1 when a recorded hash no longer matches")
    parser.add_argument("--reason", default="unspecified input-byte change",
                        help="Why the hashes are being re-recorded (printed in the log)")
    parser.add_argument("--root", type=Path, default=Path(config.ARTIFACTS_DIR))
    args = parser.parse_args(argv)

    changed_any = False
    for name in ARTIFACTS:
        changed, notes = process(args.root / name, check=args.check, reason=args.reason)
        changed_any = changed_any or changed
        for line in notes:
            print(("STALE " if args.check and changed else "OK    " if not changed else "UPDATE ") + line)

    if args.check:
        print("source hashes: " + ("MISMATCH — artifacts are stale" if changed_any else "all match"))
        return 1 if changed_any else 0
    print("source hashes: " + ("updated" if changed_any else "nothing to do"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
