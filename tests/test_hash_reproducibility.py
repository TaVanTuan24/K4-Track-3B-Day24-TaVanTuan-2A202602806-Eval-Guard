"""Regression tests for cross-platform hash reproducibility and artifact consistency.

The bug these pin down: with no `.gitattributes` and `core.autocrlf=true`, git stored LF
blobs but checked CRLF into the working tree, while `metadata.source_sha256` hashed the
WORKING-TREE bytes via `Path.read_bytes()`. The artifacts therefore passed
`check_lab.py` on the Windows machine that produced them and failed on any Linux/CI
clone, where the working tree is LF. These tests fail if either half of the fix regresses:
the normalization (`.gitattributes` + LF working tree) or the recorded hashes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
from src.report_contracts import validate_metadata
from src.experiment_support import read_json, write_json

pytestmark = pytest.mark.support

MODEL_ROLES = ("generation", "evaluator", "judge", "guard")

ARTIFACTS = ("ragas_50q.json", "judge_results.json", "guard_results.json")

SOURCE_FILES = (
    config.ANSWERS_PATH,
    config.TEST_SET_PATH,
    config.HUMAN_LABELS_PATH,
    config.ADVERSARIAL_SET_PATH,
    config.CLEAN_SET_PATH,
    config.OUTPUT_SET_PATH,
)


# ---------------------------------------------------------------------------
# The recorded hashes must match the real bytes, on any checkout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_recorded_source_hashes_match_real_bytes(artifact):
    """Every recorded hash equals the SHA-256 of the file as checked out."""
    import hashlib

    report = read_json(Path(config.ARTIFACTS_DIR) / artifact)
    recorded = report["metadata"]["source_sha256"]
    assert recorded, f"{artifact}: metadata.source_sha256 is empty"
    for name, digest in recorded.items():
        path = next((Path(p) for p in SOURCE_FILES if Path(p).name == name), None)
        assert path is not None, f"{artifact}: unknown source {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, (
            f"{artifact}: recorded hash for {name} does not match the file on disk. "
            "Line endings probably differ from the committed (LF) form — see .gitattributes.")


def test_source_files_are_checked_out_with_lf():
    """No CRLF in any hashed source: the invariant .gitattributes enforces.

    If this fails, the working tree was written by a tool that ignores .gitattributes
    (or the attribute rules were weakened), and every recorded hash becomes
    platform-dependent again.
    """
    offenders = [Path(p).name for p in SOURCE_FILES if b"\r\n" in Path(p).read_bytes()]
    assert not offenders, f"CRLF found in hashed sources: {offenders}"


def test_json_writers_emit_lf_only(tmp_path):
    """The writers themselves must be platform-independent, not just .gitattributes.

    This is the deeper half of the CRLF bug: in text mode Python translates "\\n" to
    `os.linesep`, so on Windows every artifact and every generated answers file was
    written as CRLF. Since `metadata.source_sha256` hashes raw bytes, re-running
    setup_answers on Windows would then record a hash that a Linux clone cannot match.
    """
    from src.checkpoint import atomic_write_json
    from src.experiment_support import write_json

    payload = {"vi": "nghỉ phép năm", "nested": {"list": [1, 2, 3], "empty": None}}
    atomic = tmp_path / "atomic.json"
    helper = tmp_path / "helper.json"
    atomic_write_json(atomic, payload)
    write_json(helper, payload)

    for path in (atomic, helper):
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"{path.name} must be written with LF on every platform"
        assert raw.endswith(b"\n"), f"{path.name} must end with a single LF"
    assert atomic.read_bytes() == helper.read_bytes(), "both writers must agree byte for byte"


def test_gitattributes_pins_lf_and_protects_binaries():
    path = Path(config.ROOT_DIR) / ".gitattributes"
    assert path.is_file(), ".gitattributes is missing: hashes become platform-dependent"
    text = path.read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text, "the catch-all LF rule is missing"
    for pattern in ("*.json", "*.md", "*.py", "*.yml", "*.txt"):
        assert f"{pattern}" in text and "eol=lf" in text, f"{pattern} is not pinned to LF"
    assert "*.pdf" in text and "binary" in text, "binary payloads must not be converted"


def test_validator_rejects_a_crlf_rendering_of_a_source(tmp_path):
    """The exact original failure: a CRLF checkout must be detected as stale evidence."""
    report = read_json(Path(config.ARTIFACTS_DIR) / "guard_results.json")
    source = Path(config.CLEAN_SET_PATH)
    crlf = tmp_path / source.name
    crlf.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="Stale experiment source"):
        validate_metadata(report, [crlf])


def test_validator_rejects_any_modified_source(tmp_path):
    """A one-character edit is enough to invalidate recorded evidence."""
    report = read_json(Path(config.ARTIFACTS_DIR) / "ragas_50q.json")
    modified = tmp_path / Path(config.TEST_SET_PATH).name
    payload = read_json(config.TEST_SET_PATH)
    payload[0]["question"] = payload[0]["question"] + " (edited)"
    write_json(modified, payload)
    with pytest.raises(ValueError, match="Stale experiment source"):
        validate_metadata(report, [Path(config.ANSWERS_PATH), modified])


# ---------------------------------------------------------------------------
# The metadata-only refresh tool
# ---------------------------------------------------------------------------


def test_refresh_tool_check_passes_on_committed_artifacts():
    from scripts import refresh_metadata_hashes

    assert refresh_metadata_hashes.main(["--check"]) == 0


def test_refresh_tool_repairs_hashes_without_touching_anything_else(tmp_path):
    from scripts import refresh_metadata_hashes

    source = Path(config.ARTIFACTS_DIR) / "ragas_50q.json"
    target = tmp_path / "ragas_50q.json"
    payload = read_json(source)
    payload["metadata"]["source_sha256"]["answers_50q.json"] = "0" * 64
    write_json(target, payload)

    changed, _ = refresh_metadata_hashes.process(target, check=False, reason="test")
    assert changed is True

    repaired = read_json(target)
    assert repaired["metadata"]["source_sha256"]["answers_50q.json"] != "0" * 64
    # Everything except the hash map is identical.
    for key in ("per_question", "per_distribution", "failure_clusters", "bottom_10",
                "total_questions"):
        assert repaired[key] == payload[key], f"{key} must not be touched"
    for key in ("format_version", "run_kind", "created_at", "models"):
        assert repaired["metadata"][key] == payload["metadata"][key]


def test_refresh_tool_refuses_unknown_source_names():
    from scripts import refresh_metadata_hashes

    with pytest.raises(SystemExit, match="Unknown source file"):
        refresh_metadata_hashes.recompute({"definitely_not_a_source.json": "0" * 64})


def test_annotation_tool_check_passes_and_is_idempotent():
    from scripts import refresh_phase_a_annotations

    assert refresh_phase_a_annotations.main(["--check"]) == 0
    # Running the export again is a no-op rather than a rewrite.
    assert refresh_phase_a_annotations.main([]) == 0


def test_annotation_tool_refuses_to_rewrite_ranking():
    from scripts import refresh_phase_a_annotations

    stored = read_json(Path(config.ARTIFACTS_DIR) / "ragas_50q.json")["bottom_10"]
    tampered = [dict(row) for row in stored]
    tampered[0]["rank"] = 99
    with pytest.raises(SystemExit, match="Refusing to write"):
        refresh_phase_a_annotations.verify_identity(tampered, stored)


def test_annotation_tool_rejects_incomplete_bottom10():
    from scripts import refresh_phase_a_annotations

    stored = read_json(Path(config.ARTIFACTS_DIR) / "ragas_50q.json")["bottom_10"]
    with pytest.raises(SystemExit, match="refusing"):
        refresh_phase_a_annotations.verify_identity(stored[:-1], stored)


# ---------------------------------------------------------------------------
# Artifact ↔ schema/model consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_metadata_contract_is_live_v2(artifact):
    report = read_json(Path(config.ARTIFACTS_DIR) / artifact)
    meta = report["metadata"]
    assert meta["format_version"] == 2
    assert meta["run_kind"] == "live"
    assert set(meta["models"]) == set(MODEL_ROLES)
    for role, models in meta["models"].items():
        assert isinstance(models, list)
        assert all(isinstance(m, str) and m.strip() for m in models), f"{role}: blank model id"
        assert models == sorted(set(models)), f"{role}: models must be sorted and unique"


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_no_placeholder_or_fake_model_ids(artifact):
    """Guard against invented provenance: ids must look like real provider ids."""
    report = read_json(Path(config.ARTIFACTS_DIR) / artifact)
    banned = ("test", "mock", "fake", "dummy", "todo", "xxx", "placeholder", "example")
    for role, models in report["metadata"]["models"].items():
        for model in models:
            lowered = model.lower()
            assert not any(word in lowered for word in banned), f"{role}: suspicious model id {model!r}"
            assert "/" in model or lowered.startswith(("gpt", "claude", "gemini")), (
                f"{role}: {model!r} does not look like a provider model id")


def test_answers_artifact_matches_the_test_set_exactly():
    answers = read_json(config.ANSWERS_PATH)
    questions = read_json(config.TEST_SET_PATH)
    assert len(answers) == 50
    assert [row["id"] for row in answers] == [row["id"] for row in questions]
    by_id = {q["id"]: q for q in questions}
    for row in answers:
        original = by_id[row["id"]]
        for field in ("question", "distribution", "ground_truth"):
            assert row[field] == original[field], f"id {row['id']}: {field} drifted"
        assert row["answer"].strip() != row["ground_truth"].strip(), "answer must not be the reference"


def test_phase_a_rows_align_with_answers_and_metrics_are_finite():
    report = read_json(Path(config.ARTIFACTS_DIR) / "ragas_50q.json")
    answers = read_json(config.ANSWERS_PATH)
    rows = report["per_question"]
    assert len(rows) == 50
    assert [r["question_id"] for r in rows] == [a["id"] for a in answers]
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    for row in rows:
        for name in metrics:
            value = row[name]
            assert isinstance(value, float) and 0.0 <= value <= 1.0, f"id {row['question_id']}: {name}"
    # `worst_metric` is a derived property (not serialized per row), so it is checked
    # against the rows where it IS recorded.
    by_id = {r["question_id"]: r for r in rows}
    for bottom in report["bottom_10"]:
        source = by_id[bottom["question_id"]]
        assert bottom["worst_metric"] == min(metrics, key=lambda m: source[m])
    # Per-distribution averages must match a fresh recomputation.
    for distribution, summary in report["per_distribution"].items():
        subset = [r for r in rows if r["distribution"] == distribution]
        assert summary["count"] == len(subset)
        for name in metrics:
            expected = sum(r[name] for r in subset) / len(subset)
            assert summary[name] == pytest.approx(expected, abs=1e-6), f"{distribution}.{name}"


def test_phase_a_bottom10_is_a_sorted_subset_of_the_weakest_rows():
    report = read_json(Path(config.ARTIFACTS_DIR) / "ragas_50q.json")
    rows = report["per_question"]
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    expected = sorted(sum(r[m] for m in metrics) / 4 for r in rows)[:10]
    bottom = report["bottom_10"]
    assert [r["rank"] for r in bottom] == list(range(1, 11))
    # avg_score is rounded to 4 decimals in the artifact; the validator uses the same
    # 1e-4 tolerance, so mirror it here instead of demanding byte equality.
    for row, want in zip(bottom, expected):
        assert row["avg_score"] == pytest.approx(want, abs=1e-4)
    assert len({r["question_id"] for r in bottom}) == 10
    for row in bottom:
        assert row["diagnosis"].strip() and row["suggested_fix"].strip()


def test_phase_a_cluster_matrix_covers_every_row():
    report = read_json(Path(config.ARTIFACTS_DIR) / "ragas_50q.json")
    rows = report["per_question"]
    matrix = report["failure_clusters"]["matrix"]
    counted = sum(sum(dist.values()) for dist in matrix.values())
    assert counted == len(rows)
    for distribution in ("factual", "multi_hop", "adversarial"):
        total = sum(matrix[metric][distribution] for metric in matrix)
        assert total == sum(1 for r in rows if r["distribution"] == distribution)


def test_phase_b_uses_the_prescribed_pair_sources():
    report = read_json(Path(config.ARTIFACTS_DIR) / "judge_results.json")
    assert [r["question_id"] for r in report["pairwise_results"]] == list(config.PAIR_QUESTION_IDS)
    assert len(report["binary_judgments"]) == 10
    answers = {a["id"]: a for a in read_json(config.ANSWERS_PATH)}
    humans = {h["question_id"]: h for h in read_json(config.HUMAN_LABELS_PATH)}
    for row in report["pairwise_results"]:
        assert row["answer_a"] == answers[row["question_id"]]["answer"]
        assert row["answer_b"] == humans[row["question_id"]]["model_answer"]
    for row in report["binary_judgments"]:
        assert row["model_answer"] == humans[row["question_id"]]["model_answer"]
        assert type(row["label"]) is int and row["label"] in (0, 1)
        assert "human_label" not in json.dumps(row) and "human_note" not in json.dumps(row)


def test_phase_c_counts_and_latency_scope():
    report = read_json(Path(config.ARTIFACTS_DIR) / "guard_results.json")
    assert report["adversarial"]["total"] == 20
    assert report["clean"]["total"] == 8
    assert len(report["output_results"]) == 4
    latency = report["latency"]
    assert latency["scope"] == "input_layers_sequential"
    assert latency["percentile_method"] == "linear"
    assert latency["n_samples"] == len(latency["samples"]) >= 20
    assert latency["warmup_runs"] >= 2
