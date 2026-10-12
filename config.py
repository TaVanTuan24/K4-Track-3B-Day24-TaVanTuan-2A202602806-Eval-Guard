"""Shared configuration for K4 / Track 3B: Eval + Guardrail Stack.

Provider: Google Gemini through its OpenAI-compatible endpoint (free tier).
The same settings also work against any other OpenAI-compatible endpoint by
setting OPENAI_BASE_URL and GEMINI_API_KEY/OPENAI_API_KEY in .env.

Only variable *names* belong in .env.example; real keys stay in .env, which is
git-ignored. Nothing here may hardcode a secret.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    """Read a string setting; blank/whitespace values fall back to the default."""
    value = os.getenv(name)
    return value.strip() if value is not None and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


# --- API keys ---
# Gemini free tier is the target provider. OPENAI_API_KEY stays supported (Day 18
# modules and the supplied tests import that name), and the two are interchangeable:
# the first non-empty value wins.
GEMINI_API_KEY = _env("GEMINI_API_KEY")
OPENAI_API_KEY = _env("OPENAI_API_KEY") or GEMINI_API_KEY
API_KEY = OPENAI_API_KEY  # canonical credential used by the shared client factory

HF_TOKEN = _env("HF_TOKEN")  # Optional: for HuggingFace models

# --- OpenAI-compatible endpoint ---
# Gemini: https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", GEMINI_BASE_URL)

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = _env("QDRANT_HOST", "localhost")
QDRANT_PORT = _env_int("QDRANT_PORT", 6333)
COLLECTION_NAME = "lab24_production"
NAIVE_COLLECTION = "lab24_naive"

# --- Embedding (local; no API quota) ---
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024)

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = _env_int("HIERARCHICAL_PARENT_SIZE", 2048)
HIERARCHICAL_CHILD_SIZE = _env_int("HIERARCHICAL_CHILD_SIZE", 256)
SEMANTIC_THRESHOLD = _env_float("SEMANTIC_THRESHOLD", 0.85)

# --- Search (same as Day 18) ---
BM25_TOP_K = _env_int("BM25_TOP_K", 20)
DENSE_TOP_K = _env_int("DENSE_TOP_K", 20)
HYBRID_TOP_K = _env_int("HYBRID_TOP_K", 20)
RERANK_TOP_K = _env_int("RERANK_TOP_K", 3)
# flashrank cross-encoder. MultiBERT-L-12 is the multilingual MS MARCO reranker, the
# only sensible default for a Vietnamese corpus; flashrank's own default (TinyBERT-L-2)
# cannot separate near-duplicate policy versions (v2023 vs v2024).
RERANK_MODEL = _env("RERANK_MODEL", "ms-marco-MultiBERT-L-12")
# If the cross-encoder's top-1 and last score differ by less than this, it is not
# discriminating on this corpus (off-domain saturation) and the hybrid order is kept.
RERANK_MIN_SPREAD = _env_float("RERANK_MIN_SPREAD", 1e-3)
# Saturation is also visible as a compressed HIGH band: when even the worst candidate
# scores >= this, the model is calling everything nearly perfectly relevant. Measured
# on this corpus with ms-marco-MultiBERT-L-12: min=0.996963, max=0.999552 over 20
# candidates — a spread (2.59e-3) above the threshold above, yet the ordering it
# produced put a "thưởng Tết" chunk above the correct "nghỉ phép đặc biệt" chunk.
RERANK_SATURATION_FLOOR = _env_float("RERANK_SATURATION_FLOOR", 0.95)

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FIXTURES_DIR = os.path.join(ROOT_DIR, "fixtures")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
TEST_SET_PATH = os.path.join(FIXTURES_DIR, "test_set_50q.json")
ANSWERS_PATH = os.path.join(ARTIFACTS_DIR, "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(FIXTURES_DIR, "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(FIXTURES_DIR, "adversarial_set_20.json")
CLEAN_SET_PATH = os.path.join(FIXTURES_DIR, "clean_set_8.json")
OUTPUT_SET_PATH = os.path.join(FIXTURES_DIR, "output_set_4.json")
GUARDRAILS_CONFIG_DIR = os.path.join(ROOT_DIR, "guardrails")
PAIR_QUESTION_IDS = (1, 5, 12, 21, 41)

# --- Models ---
# gemini-3.5-flash-lite is the preferred free-tier model; override in .env when the
# account does not expose it (run scripts/check_api.py to list what is available).
DEFAULT_GENERATION_MODEL = "gemini-3.5-flash-lite"
GENERATION_MODEL = _env("GENERATION_MODEL", DEFAULT_GENERATION_MODEL)
JUDGE_MODEL = _env("JUDGE_MODEL", GENERATION_MODEL)
# m4/RAGAS evaluator LLM. Kept separate so evaluation can use a different model
# than generation or judging, and reported through metadata.evaluator_models.
EVALUATOR_MODEL = _env("EVALUATOR_MODEL", JUDGE_MODEL)

# --- Retry / rate limiting (free-tier friendly) ---
# 429 and 5xx responses are retried with exponential backoff + jitter.
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 6)
LLM_RETRY_BASE_DELAY_S = _env_float("LLM_RETRY_BASE_DELAY_S", 4.0)
LLM_RETRY_MAX_DELAY_S = _env_float("LLM_RETRY_MAX_DELAY_S", 90.0)
LLM_TIMEOUT_S = _env_float("LLM_TIMEOUT_S", 120.0)
# Proactive pacing: minimum seconds between successive API calls. Live entry points
# enable this explicitly (never during offline unit tests).
LLM_MIN_INTERVAL_S = _env_float("LLM_MIN_INTERVAL_S", 4.0)
# Concurrent in-flight requests. Free tier works best with a low value.
LLM_MAX_CONCURRENCY = _env_int("LLM_MAX_CONCURRENCY", 2)

# --- m5 enrichment (optional LLM pass over chunks) ---
# Enrichment spends one call per batch of chunks. It is real API work, so it can
# be switched off when the free-tier budget must be reserved for the phases.
ENRICH_ENABLED = _env("ENRICH_ENABLED", "1").lower() not in ("0", "false", "no")
ENRICH_BATCH_SIZE = _env_int("ENRICH_BATCH_SIZE", 8)

# --- Phase A evaluation (m4) ---
# auto: use the ragas library, fall back to the built-in four-metric
# implementation if the library is incompatible with the endpoint; library /
# native force one path. Both paths make real API calls.
RAGAS_IMPL = _env("RAGAS_IMPL", "auto").lower()
RAGAS_MAX_WORKERS = _env_int("RAGAS_MAX_WORKERS", 1)
RAGAS_STRICTNESS = _env_int("RAGAS_STRICTNESS", 1)

# --- Resumable progress caches (never re-spend succeeded calls) ---
CACHE_DIR = os.path.join(ARTIFACTS_DIR, "cache")
ENRICHMENT_CACHE_PATH = os.path.join(CACHE_DIR, "enrichment.json")
RAGAS_CACHE_PATH = os.path.join(CACHE_DIR, "ragas.json")
ANSWERS_PROGRESS_PATH = os.path.join(CACHE_DIR, "answers_50q_progress.json")

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # Full guard target; Task 12 currently measures input guards only.
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer
