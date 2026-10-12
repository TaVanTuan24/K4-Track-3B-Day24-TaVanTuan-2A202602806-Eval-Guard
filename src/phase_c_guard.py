from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency.

The NeMo instance is configured in guardrails/config.yml against the same
OpenAI-compatible endpoint as the rest of the project (Gemini by default), so the
input/output self-check rails are real LLM calls rather than keyword heuristics.
Public entry points run in one event loop per stage: an HTTP client bound to a
closed loop cannot be reused across stages.
"""

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as app_config
import src.llm_client as llm
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

# Entities that count as PII for this assistant: the identifiers Task 9a must catch
# (Vietnamese CCCD/CMND and phone, email) plus the international equivalents.
# Everything else Presidio can return (DATE_TIME, NRP, LOCATION, PERSON, URL, ...) is
# not blocking-worthy here and is deliberately excluded — see pii_scan().
PII_ENTITIES = frozenset({
    "VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "IBAN_CODE", "US_SSN",
})


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    # Keep only identifiers that are actually PII for this assistant. Presidio's
    # predefined recognizers also return DATE_TIME, NRP, LOCATION, PERSON, URL, ...
    # Treating every entity type as PII blocked ALL EIGHT clean controls on the real
    # engine (measured fpr=0.875): "chính sách v2024" matched DATE_TIME and
    # "ngày mai" matched DATE. The filter is applied to the results instead of passing
    # `entities=` to analyze() so the engine call stays the same shape.
    results = [r for r in results if r.entity_type in PII_ENTITIES]
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {"type": r.entity_type, "text": text[r.start:r.end],
         "score": round(r.score, 3), "start": r.start, "end": r.end}
        for r in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def register_gemini_engine():
    """Register `engine: gemini` as langchain_openai.ChatOpenAI for NeMo 0.10.0.

    Verified against the installed 0.10.0 source
    (nemoguardrails/llm/providers/providers.py::get_llm_provider): with
    `engine: openai`, the provider class is chosen by *model name substring* —
    names containing "gpt-3.5"/"gpt-4" get ChatOpenAI, everything else gets
    langchain_openai.OpenAI (the legacy /completions LLM), which Gemini's
    OpenAI-compatible endpoint does not implement. Registering a dedicated engine
    is the version-correct way to get a chat model on 0.10.0 — the newer
    flow/library syntax does not apply here.
    """
    from langchain_openai import ChatOpenAI
    from nemoguardrails.llm.providers.providers import register_llm_provider
    register_llm_provider("gemini", ChatOpenAI)
    return ChatOpenAI


def _apply_provider_settings(rails_config) -> None:
    """Push .env endpoint/key/model into the NeMo model config.

    RailsConfig YAML on 0.10.0 has no environment interpolation, so config.yml
    carries documented placeholders and the values are injected here. `model` is
    set in `parameters` explicitly: NeMo only forwards the model name as
    `kwargs["model"]` when the provider class exposes a field literally named
    `model` (ChatOpenAI exposes `model_name`, with `model` as an alias), so relying
    on that lookup alone would silently fall back to the SDK default model.
    """
    api_key = llm.resolve_api_key()
    if not api_key:
        raise llm.MissingCredential(llm.credential_hint())
    for model in getattr(rails_config, "models", []) or []:
        if model.engine != "gemini":
            continue
        parameters = dict(model.parameters or {})
        parameters["model"] = app_config.GENERATION_MODEL
        parameters["base_url"] = llm.resolve_base_url()
        parameters["api_key"] = api_key
        parameters["timeout"] = app_config.LLM_TIMEOUT_S
        model.parameters = parameters
        model.model = app_config.GENERATION_MODEL


def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Gemini-backed)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — câu từ chối Colang 1.0; built-in self-check flows được bật trong YAML

    The NeMo instance declares its LLM in `rails.config.models`, which is what
    Phase C metadata reports as the guard model.
    """
    from nemoguardrails import RailsConfig, LLMRails
    register_gemini_engine()
    rails_config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    _apply_provider_settings(rails_config)
    rails = LLMRails(rails_config)
    return rails


def _activated_rails_decision(result) -> bool:
    """Read NeMo's own rail decision instead of guessing from refusal wording."""
    log = getattr(result, "log", None)
    activated = getattr(log, "activated_rails", None) or []
    return any(getattr(rail, "stop", False) for rail in activated)


def _response_text(result) -> str:
    response = getattr(result, "response", None)
    if isinstance(response, list) and response:
        first = response[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
        return str(first)
    return str(response)


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    result = await rails.generate_async(
        messages=[{"role": "user", "content": text}],
        options={"rails": ["input"], "log": {"activated_rails": True}},
    )
    response = _response_text(result)
    blocked = _activated_rails_decision(result)
    return {
        "allowed":        not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response":       response,
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails chạy trong context của cả hội thoại (input + output). Self-check
    này không thay RAGAS faithfulness hoặc kiểm tra factual accuracy với contexts.

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is None:
        rails = setup_nemo_rails()

    result = await rails.generate_async(
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        options={"rails": ["output"], "log": {"activated_rails": True}},
    )
    response = _response_text(result)
    flagged = _activated_rails_decision(result)
    return {
        "safe":           not flagged,
        "flagged_reason": "nemo_output_rail" if flagged else None,
        "final_answer":   response if flagged else answer,
    }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    `expected` is only used to score the outcome, never to decide it.

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII (synchronous, fast)
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail (async — await, không dùng asyncio.run())
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      item["input"],
                "expected":   item["expected"],
                "actual":     actual,
                "blocked_by": blocked_by,
                "passed":     actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())   # một lần duy nhất — không gọi asyncio.run() trong loop
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: measure INPUT layers sequentially, not end-to-end guard/RAG latency.

    Two real warm-up rounds per layer (excluded from the samples), then exactly
    n_runs timed iterations cycling the clean inputs. Both layers are always timed,
    even when Presidio already flags the text — no short-circuit, because the point
    is to measure each layer's cost independently.
    """
    from src.experiment_support import summarize_latency
    if not test_inputs or n_runs < 1:
        raise ValueError("Need inputs and n_runs >= 1")
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    if rails is None:
        rails = setup_nemo_rails()

    async def _measure():
        for i in range(2):
            text = test_inputs[i % len(test_inputs)]
            pii_scan(text, analyzer, anonymizer)
            await check_input_rail(text, rails)
        samples = []
        for i in range(n_runs):
            text = test_inputs[i % len(test_inputs)]
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            pii_ms = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000
            samples.append({"presidio_ms": pii_ms, "nemo_input_ms": nemo_ms,
                            "input_guard_ms": pii_ms + nemo_ms})
        return samples

    return summarize_latency(asyncio.run(_measure()), warmup_runs=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "c", *sys.argv[1:]]))
