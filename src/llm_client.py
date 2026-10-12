"""Shared OpenAI-compatible client factory + retry/backoff for the whole project.

Every LLM call in Day 24 goes through here: answer generation (`setup_answers`),
the Phase A m4/RAGAS evaluator, the Phase B pairwise judge, the supplied binary
0/1 judge, the Day 18 pipeline modules (m2 dense embeddings are local, m4/m5 call
the LLM) and — through `guardrails/config.yml` — NeMo Guardrails.

Design notes
------------
* `get_client()` resolves the endpoint from `config.OPENAI_BASE_URL` (Gemini's
  OpenAI-compatible surface by default) and the credential from `config.API_KEY`.
  The SDK is imported lazily so offline tests that monkeypatch `sys.modules`
  keep intercepting the constructor.
* Clients are cached so one process reuses a single HTTP connection pool, but the
  cache entry also records the `OpenAI` callable it was built from. If a test (or
  a caller) swaps the SDK object, the stale client is dropped instead of leaking
  across tests.
* `call_with_retry()` retries only transient failures (HTTP 429/5xx, timeouts,
  connection errors) with exponential backoff + jitter, honouring `Retry-After`.
  A permanent error is raised immediately — never swallowed into fake data.
* Pacing (`set_pacing`) is opt-in and enabled by the live entry points only, so
  unit tests never sleep.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any, Callable

import config

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MissingCredential(RuntimeError):
    """Raised when a live call is attempted without a configured API key."""


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

_CLIENTS: dict[tuple[str, str], tuple[Any, Any]] = {}
_CLIENTS_LOCK = threading.Lock()


def resolve_base_url(base_url: str | None = None) -> str:
    value = (base_url or config.OPENAI_BASE_URL or "").strip()
    return value or config.GEMINI_BASE_URL


def resolve_api_key(api_key: str | None = None) -> str:
    if api_key is not None:
        return api_key.strip()
    return (config.API_KEY or "").strip()


def get_client(*, role: str = "llm", base_url: str | None = None,
               api_key: str | None = None) -> Any:
    """Return a cached OpenAI-compatible client for `role`.

    `role` is diagnostic only (`generation`/`evaluator`/`judge`/`guard`); it does
    not select the model, because callers pass `model=` explicitly. Callers that
    need one client for many calls (the 50-question loop) should call this once
    and reuse the returned object.
    """
    endpoint = resolve_base_url(base_url)
    key = resolve_api_key(api_key)

    # Lazy import: offline tests replace sys.modules["openai"] with a fake module.
    from openai import OpenAI  # noqa: WPS433 (intentional deferred import)

    cache_key = (endpoint, key)
    with _CLIENTS_LOCK:
        cached = _CLIENTS.get(cache_key)
        if cached is not None and cached[0] is OpenAI:
            return cached[1]

    kwargs: dict[str, Any] = {"base_url": endpoint, "timeout": config.LLM_TIMEOUT_S}
    if key:
        kwargs["api_key"] = key
    client = OpenAI(**kwargs)

    with _CLIENTS_LOCK:
        _CLIENTS[cache_key] = (OpenAI, client)
    return client


def reset_client_cache() -> None:
    """Drop cached clients (used by tests and by the live CLIs between stages)."""
    with _CLIENTS_LOCK:
        _CLIENTS.clear()


def credential_hint() -> str:
    """Human-readable diagnosis for CLI entry points; never prints the key."""
    if resolve_api_key():
        return f"credential configured for {resolve_base_url()}"
    return ("No API key found. Put GEMINI_API_KEY=<key> in .env "
            "(see .env.example). Never commit .env or paste the key into source.")


def require_api_key() -> str:
    """Return the credential or raise MissingCredential with an actionable message."""
    key = resolve_api_key()
    if not key:
        raise MissingCredential(credential_hint())
    return key


# ---------------------------------------------------------------------------
# Pacing (opt-in) — free-tier rate limiting
# ---------------------------------------------------------------------------

_PACE_LOCK = threading.Lock()
_LAST_CALL_AT = 0.0
_MIN_INTERVAL_S = 0.0


def set_pacing(min_interval_s: float | None) -> None:
    """Enable/disable proactive spacing between requests.

    Live entry points call `set_pacing(config.LLM_MIN_INTERVAL_S)`. Tests never
    do, so `_pace()` is a no-op there and the suite stays fast.
    """
    global _MIN_INTERVAL_S
    with _PACE_LOCK:
        _MIN_INTERVAL_S = max(0.0, float(min_interval_s)) if min_interval_s else 0.0


def _pace() -> None:
    global _LAST_CALL_AT
    if _MIN_INTERVAL_S <= 0:
        return
    with _PACE_LOCK:
        now = time.monotonic()
        wait = _MIN_INTERVAL_S - (now - _LAST_CALL_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_AT = time.monotonic()


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRYABLE_NAME_PARTS = (
    "ratelimit", "apiconnection", "apitimeout", "internalserver",
    "serviceunavailable", "timeout", "connectionerror", "resourceexhausted",
)
_RETRYABLE_TEXT_PARTS = (
    "429", "rate limit", "ratelimit", "resource_exhausted", "resource exhausted",
    "quota", "overloaded", "temporarily unavailable", "timed out", "timeout",
    "connection reset", "connection aborted", "502", "503", "504",
)


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable(exc: BaseException) -> bool:
    """Transient-failure test that works with the real SDK and with test doubles."""
    status = _status_code(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    if any(part in name for part in _RETRYABLE_NAME_PARTS):
        return True
    return any(part in str(exc).lower() for part in _RETRYABLE_TEXT_PARTS)


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return None


def call_with_retry(fn: Callable[[], Any], *, max_retries: int | None = None,
                    label: str = "llm call", on_retry: Callable[[int, float, BaseException], None] | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> Any:
    """Run `fn()`; retry only transient failures with exponential backoff + jitter."""
    attempts = config.LLM_MAX_RETRIES if max_retries is None else max(0, max_retries)
    for attempt in range(attempts + 1):
        _pace()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised unless transient
            if attempt >= attempts or not is_retryable(exc):
                raise
            delay = min(config.LLM_RETRY_BASE_DELAY_S * (2 ** attempt),
                        config.LLM_RETRY_MAX_DELAY_S)
            delay += random.uniform(0, min(2.0, delay * 0.25))
            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
            if on_retry is not None:
                on_retry(attempt + 1, delay, exc)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(?P<body>.*?)\s*```", re.DOTALL)


def parse_json_object(content: str, *, context: str = "LLM response") -> dict:
    """Parse a JSON object out of raw model text.

    Handles bare JSON, fenced ```json blocks and prose-wrapped objects. Raises
    ValueError (never returns an invented object) when nothing parses.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{context}: empty content, expected a JSON object.")
    text = content.strip()

    candidates: list[str] = [text]
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group("body").strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    snippet = text[:200].replace("\n", " ")
    raise ValueError(f"{context}: could not parse a JSON object from: {snippet!r}")


def _looks_like_response_format_rejection(exc: BaseException) -> bool:
    text = str(exc).lower()
    status = _status_code(exc)
    if status not in (400, 404, 422):
        return False
    return ("response_format" in text or "responseformat" in text
            or "json_object" in text or "json mode" in text)


def chat_json(client: Any, *, model: str, messages: list[dict], temperature: float = 0.0,
              max_retries: int | None = None, json_attempts: int = 3,
              label: str = "json call", on_retry: Callable[[int, float, BaseException], None] | None = None) -> dict:
    """Call chat completions asking for a JSON object and return the parsed dict.

    Two failure modes are handled explicitly:
      * transient transport errors -> `call_with_retry`
      * unparseable content -> the call is repeated (up to `json_attempts`)
    If the endpoint rejects `response_format`, the request is re-issued without it
    and the response is still parsed strictly.
    """
    last_error: Exception | None = None
    use_response_format = True
    for _ in range(max(1, json_attempts)):
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if use_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = call_with_retry(
                lambda kwargs=kwargs: client.chat.completions.create(**kwargs),
                max_retries=max_retries, label=label, on_retry=on_retry)
        except Exception as exc:  # noqa: BLE001
            if kwargs.get("response_format") and _looks_like_response_format_rejection(exc):
                use_response_format = False
                continue
            raise
        content = response.choices[0].message.content
        try:
            return parse_json_object(content, context=f"{label} ({model})")
        except ValueError as exc:
            last_error = exc
            continue
    raise ValueError(f"{label}: no parseable JSON object after {json_attempts} attempts: {last_error}")


def chat_text_response(client: Any, *, model: str, messages: list[dict], temperature: float = 0.0,
                       max_retries: int | None = None, label: str = "chat call",
                       on_retry: Callable[[int, float, BaseException], None] | None = None) -> tuple[str, str]:
    """Plain text completion returning (content, served_model_id).

    The served id is what the provider reports (Gemini may resolve an alias to a
    concrete snapshot); it falls back to the requested id when absent.
    """
    response = call_with_retry(
        lambda: client.chat.completions.create(model=model, messages=messages, temperature=temperature),
        max_retries=max_retries, label=label, on_retry=on_retry)
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{label}: empty completion from {model}.")
    return content, reported_model(response, model)


def chat_text(client: Any, *, model: str, messages: list[dict], temperature: float = 0.0,
              max_retries: int | None = None, label: str = "chat call",
              on_retry: Callable[[int, float, BaseException], None] | None = None) -> str:
    """Plain text completion (content only)."""
    return chat_text_response(client, model=model, messages=messages, temperature=temperature,
                              max_retries=max_retries, label=label, on_retry=on_retry)[0]


def reported_model(response: Any, fallback: str) -> str:
    """Model id the provider says it served, else the id we requested."""
    value = getattr(response, "model", None)
    return value if isinstance(value, str) and value.strip() else fallback


# ---------------------------------------------------------------------------
# Model discovery (smoke test / quota diagnosis)
# ---------------------------------------------------------------------------


def list_models(client: Any | None = None) -> list[str]:
    """Return the model ids the configured endpoint exposes."""
    client = client or get_client(role="discovery")
    ids: list[str] = []
    response = call_with_retry(lambda: client.models.list(), label="models list")
    for item in getattr(response, "data", []) or []:
        model_id = getattr(item, "id", None)
        if isinstance(model_id, str) and model_id.strip():
            ids.append(model_id.strip())
    return sorted(set(ids))


def normalize_model_id(model_id: str) -> str:
    """Drop a leading `models/` prefix so ids compare across providers."""
    value = (model_id or "").strip()
    return value[len("models/"):] if value.startswith("models/") else value


def chat_completion_smoke(client: Any, *, model: str, prompt: str = "Reply with the single word: pong",
                          max_retries: int = 2) -> dict:
    """One minimal round-trip used to prove the endpoint/model/key work."""
    response = call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0),
        max_retries=max_retries, label="smoke test")
    return {
        "requested_model": model,
        "served_model": reported_model(response, model),
        "content": response.choices[0].message.content,
    }


__all__ = [
    "MissingCredential", "get_client", "reset_client_cache", "resolve_base_url",
    "resolve_api_key", "credential_hint", "require_api_key", "set_pacing",
    "call_with_retry", "is_retryable", "parse_json_object", "chat_json",
    "chat_text", "chat_text_response", "reported_model", "list_models", "normalize_model_id",
    "chat_completion_smoke",
]
