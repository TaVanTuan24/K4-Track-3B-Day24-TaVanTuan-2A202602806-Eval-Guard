"""Crash-safe, resumable progress files for the live free-tier runs.

A 50-question RAGAS batch can take tens of minutes and may stop on quota. A run
that dies at question 37 must not re-spend API calls on 1-36, and a half-written
JSON file must never be mistaken for finished evidence.

`atomic_write_json` writes to a temporary file in the same directory and swaps it
in with `os.replace`, so readers only ever see a complete document. `ResumableStore`
keeps one record per unit key and re-reads its file on construction.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def atomic_write_json(path: str | os.PathLike, payload: Any) -> Path:
    """Write `payload` as UTF-8 JSON, atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def load_json(path: str | os.PathLike, default: Any = None) -> Any:
    """Read JSON, returning `default` when the file is absent."""
    target = Path(path)
    if not target.exists():
        return default
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def quarantine(path: str | os.PathLike) -> Path | None:
    """Move an unreadable progress file aside so a fresh run can start cleanly."""
    target = Path(path)
    if not target.exists():
        return None
    index = 0
    while True:
        candidate = target.with_suffix(target.suffix + f".corrupt{index or ''}")
        if not candidate.exists():
            break
        index += 1
    os.replace(target, candidate)
    return candidate


class ResumableStore:
    """Append-only record store keyed by a stable field, safe to resume."""

    def __init__(self, path: str | os.PathLike, *, key_field: str = "id"):
        self.path = Path(path)
        self.key_field = key_field
        self._records: dict[Any, dict] = {}
        self.recovered_from = self._load()

    def _load(self) -> Path | None:
        if not self.path.exists():
            return None
        try:
            payload = load_json(self.path, [])
        except (ValueError, OSError):
            return quarantine(self.path)
        if isinstance(payload, dict):
            payload = payload.get("records", [])
        if not isinstance(payload, list):
            return quarantine(self.path)
        for row in payload:
            if isinstance(row, dict) and self.key_field in row:
                self._records[row[self.key_field]] = row
        return None

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: Any) -> bool:
        return key in self._records

    def keys(self) -> set:
        return set(self._records)

    def get(self, key: Any) -> dict | None:
        return self._records.get(key)

    def upsert(self, record: dict, *, save: bool = True) -> None:
        self._records[record[self.key_field]] = record
        if save:
            self.save()

    def extend(self, records: Iterable[dict]) -> None:
        for record in records:
            self._records[record[self.key_field]] = record
        self.save()

    def ordered(self, order: Iterable[Any]) -> list[dict]:
        """Records in the given key order, skipping keys that are not present."""
        return [self._records[key] for key in order if key in self._records]

    def drop(self, keys: Iterable[Any], *, save: bool = True) -> int:
        """Remove records by key; returns how many were removed."""
        removed = 0
        for key in list(keys):
            if key in self._records:
                self._records.pop(key, None)
                removed += 1
        if removed and save:
            self.save()
        return removed

    def find(self, predicate) -> list[dict]:
        """Records whose dict satisfies `predicate`."""
        return [record for record in self._records.values() if predicate(record)]

    def save(self) -> None:
        atomic_write_json(self.path, list(self._records.values()))
