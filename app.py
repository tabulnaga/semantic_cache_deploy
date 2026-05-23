"""
app.py

Flask application for Semantic Cache - Store & Lookup endpoints.
Designed for AWS Lambda deployment (via apig-wsgi).

Faithfully adapted from the two Beam projects:
- beam_cache_store.py  -> POST /cache/store
- beam_cache_lookup.py -> POST /cache/lookup

Uses the ORIGINAL cache/ module (wrapper.py, cross_encoder.py, config.py)
with NO modifications. Same imports, same initialization, same logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional

from flask import Flask, request, jsonify
import redis

# --- Same imports as Beam projects ---
from cache.wrapper import SemanticCacheWrapper
from cache.cross_encoder import CrossEncoder
from cache.config import config as default_config

# ---------------------------------------------------------------------------
# Configuration (same as Beam projects)
# ---------------------------------------------------------------------------
REDIS_URL: str = os.environ.get("REDIS_URL", "")
CACHE_NAME: str = os.getenv("CACHE_NAME", default_config.get("cache_name", "semantic-cache"))
SEMANTIC_DISTANCE_THRESHOLD: float = float(
    os.getenv("CACHE_DISTANCE_THRESHOLD", default_config.get("distance_threshold", 0.3))
)
TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", default_config.get("ttl_seconds", 3600)))

ENABLE_RERANKER: bool = os.getenv("ENABLE_RERANKER", "true").lower() in ("1", "true", "yes", "y")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_DISTANCE_THRESHOLD: float = float(os.getenv("RERANKER_DISTANCE_THRESHOLD", "0.12"))

ENABLE_SIGNATURE: bool = os.getenv("ENABLE_SIGNATURE", "true").lower() in ("1", "true", "yes", "y")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
SUPABASE_DB_URL: str = os.getenv("SUPABASE_DB_URL", "")

SIGNATURE_MODEL: str = os.getenv("SIGNATURE_MODEL", "gpt-4o-mini")
SIGNATURE_TIMEOUT_SECONDS: int = int(os.getenv("SIGNATURE_TIMEOUT_SECONDS", "30"))
SIGNATURE_TABLE: str = os.getenv("SIGNATURE_TABLE", "public.company_structured_values")

# SQL Agent — MotherDuck + Claude
MOTHERDUCK_TOKEN: str = os.getenv("MOTHERDUCK_TOKEN", "")
MOTHERDUCK_DATABASE: str = os.getenv("MOTHERDUCK_DATABASE", "scouting")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
SQL_AGENT_MODEL: str = os.getenv("SQL_AGENT_MODEL", "claude-opus-4-7")
SQL_AGENT_MAX_RETRIES: int = int(os.getenv("SQL_AGENT_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Globals initialised at startup (same as Beam projects)
# ---------------------------------------------------------------------------
CACHE: Optional[SemanticCacheWrapper] = None
REDIS_CLIENT: Optional[redis.Redis] = None
TABLE_COLUMNS: List[str] = []
RERANKER: Any = None
SCHEMA_CONTEXT: str = ""  # Loaded once at startup from MotherDuck


# ---------------------------------------------------------------------------
# Redis meta helpers (exact copy from Beam projects)
# ---------------------------------------------------------------------------

def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _meta_key(prompt: str) -> str:
    return f"{CACHE_NAME}:meta:{_prompt_hash(prompt)}"


def _redis() -> redis.Redis:
    if REDIS_CLIENT is None:
        raise RuntimeError("Redis client not initialised")
    return REDIS_CLIENT


def load_meta(prompt: str) -> Optional[Dict[str, Any]]:
    raw = _redis().get(_meta_key(prompt))
    if not raw:
        return None
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        return json.loads(raw)
    except Exception:
        return None


def store_filter_signature_hash(prompt: str, filter_sig_hash: str, ttl: int) -> None:
    meta = {"filter_signature_hash": filter_sig_hash}
    _redis().set(_meta_key(prompt), json.dumps(meta, ensure_ascii=False), ex=ttl)


# ---------------------------------------------------------------------------
# SQL Filter Extraction (exact copy from Beam projects)
# ---------------------------------------------------------------------------

def _extract_filters_from_sql(sql: str) -> list[dict]:
    """Best-effort extraction of simple WHERE filters from SQL."""
    if not sql:
        return []
    s = " ".join(sql.replace("\n", " ").split())
    m = re.search(r"\bwhere\b(.*?)(\border\s+by\b|\bgroup\s+by\b|\blimit\b|$)", s, flags=re.IGNORECASE)
    if not m:
        return []
    where = m.group(1)

    filters: list[dict] = []

    # Pattern 1: lower(col) = lower('val')
    for col, val in re.findall(
        r"lower\(\s*([a-zA-Z_][\w]*)\s*\)\s*=\s*lower\(\s*'([^']*)'\s*\)", where, flags=re.IGNORECASE
    ):
        filters.append({"column": col.lower(), "op": "=", "value": val.strip().lower()})

    # Pattern 2: lower(col) = 'val'
    for col, val in re.findall(
        r"lower\(\s*([a-zA-Z_][\w]*)\s*\)\s*=\s*'([^']*)'", where, flags=re.IGNORECASE
    ):
        if col.lower() not in [f["column"] for f in filters]:
            filters.append({"column": col.lower(), "op": "=", "value": val.strip().lower()})

    # Pattern 3: col = 'val'
    for col, val in re.findall(r"\b([a-zA-Z_][\w]*)\b\s*=\s*'([^']*)'", where):
        if col.lower() not in [f["column"] for f in filters] and col.lower() != "lower":
            filters.append({"column": col.lower(), "op": "=", "value": val.strip().lower()})

    # Pattern 4: col IN ('a','b',...)
    for col, vals in re.findall(r"\b([a-zA-Z_][\w]*)\b\s+in\s*\(([^)]+)\)", where, flags=re.IGNORECASE):
        items = [v.strip(" '\"") for v in vals.split(",")]
        filters.append({"column": col.lower(), "op": "in", "value": sorted([v.lower() for v in items])})

    # Normalize & dedupe
    norm = []
    seen = set()
    for f in filters:
        col = str(f.get("column", "")).strip().lower()
        op = str(f.get("op", "=")).strip().lower()
        val = f.get("value", "")
        if isinstance(val, str):
            val_n = val.strip().lower()
        elif isinstance(val, list):
            val_n = sorted([str(x).strip().lower() for x in val])
        else:
            val_n = str(val).strip().lower()

        key = json.dumps({"column": col, "op": op, "value": val_n}, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        norm.append({"column": col, "op": op, "value": val_n})

    return sorted(
        norm,
        key=lambda x: (
            x.get("column", ""),
            x.get("op", ""),
            json.dumps(x.get("value", ""), sort_keys=True, ensure_ascii=False),
        ),
    )


# ---------------------------------------------------------------------------
# Filter signature utilities (exact copy from Beam projects)
# ---------------------------------------------------------------------------

FILTER_SIG_SCHEMA = {
    "name": "filter_signature",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "table": {"type": "string"},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "column": {"type": "string"},
                        "op": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["column", "op", "value"],
                },
            },
        },
        "required": ["table", "filters"],
    },
}


def _norm(s: str) -> str:
    return str(s).strip().lower()


def canonicalize_filter_sig(sig: Dict[str, Any]) -> Dict[str, Any]:
    table = _norm(sig.get("table") or SIGNATURE_TABLE)
    filters = sig.get("filters") or []
    out_filters = []
    for f in filters:
        try:
            col = _norm(f.get("column", ""))
            op = _norm(f.get("op", ""))
            val = _norm(f.get("value", ""))
            if col and op and val != "":
                out_filters.append({"column": col, "op": op, "value": val})
        except Exception:
            continue
    out_filters.sort(key=lambda x: (x["column"], x["op"], x["value"]))
    return {"table": table, "filters": out_filters}


def filter_signature_hash(sig: Dict[str, Any]) -> str:
    payload = json.dumps(sig, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_filter_signature(question: str, columns: List[str]) -> Dict[str, Any]:
    """Build a deterministic, filter-only signature from the question using OpenAI."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is empty.")
    if not columns:
        raise RuntimeError("No columns provided for filter signature extraction.")

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    cols = ", ".join(sorted({_norm(c) for c in columns if c}))

    system = (
        "Extract ONLY the database filter intent (WHERE conditions) from the user question. "
        "Be deterministic. Use only columns from the provided list. "
        "If user says 'domain education', output filter: {column:'domain', op:'=', value:'education'}. "
        "Return JSON only, matching the schema."
    )
    user = (
        f"Table: {SIGNATURE_TABLE}\n"
        f"Available columns (lowercase): {cols}\n\n"
        f"Question: {question}\n\n"
        "Return the filter_signature JSON."
    )

    try:
        resp = client.responses.create(
            model=SIGNATURE_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_schema", "json_schema": FILTER_SIG_SCHEMA},
            timeout=SIGNATURE_TIMEOUT_SECONDS,
        )
        parsed = resp.output_parsed
        if not isinstance(parsed, dict):
            raise RuntimeError("Unexpected OpenAI parsed output type")
        sig = parsed
    except Exception:
        resp = client.chat.completions.create(
            model=SIGNATURE_MODEL,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": FILTER_SIG_SCHEMA},
            timeout=SIGNATURE_TIMEOUT_SECONDS,
        )
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned empty content for filter signature")
        sig = json.loads(content)

    return canonicalize_filter_sig(sig)


# ---------------------------------------------------------------------------
# Reranker helper (same pattern as beam_cache_lookup.py)
# ---------------------------------------------------------------------------

def rerank_candidates(query: str, candidates: List[Dict]) -> List[Dict]:
    """Rerank candidates using the CrossEncoder reranker."""
    if not candidates:
        return []
    if RERANKER is None:
        return candidates
    return RERANKER(query, candidates)


# ---------------------------------------------------------------------------
# Cache store helper (exact copy from beam_cache_store.py)
# ---------------------------------------------------------------------------

def _cache_store(c: Any, prompt: str, response: str, ttl_seconds: int) -> None:
    """Store a single (prompt, response) pair using the underlying SemanticCache."""
    underlying = getattr(c, "cache", None)
    if underlying is None or not hasattr(underlying, "store"):
        raise AttributeError("SemanticCacheWrapper.cache.store is not available")

    try:
        underlying.store(prompt=prompt, response=response, ttl_seconds=ttl_seconds)
        return
    except TypeError:
        pass
    try:
        underlying.store(prompt=prompt, response=response, ttl=ttl_seconds)
        return
    except TypeError:
        pass
    underlying.store(prompt=prompt, response=response)


# ---------------------------------------------------------------------------
# Supabase column loader (sync)
# ---------------------------------------------------------------------------

def load_columns_from_supabase() -> List[str]:
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is empty.")

    if "." in SIGNATURE_TABLE:
        schema, table = SIGNATURE_TABLE.split(".", 1)
    else:
        schema, table = "public", SIGNATURE_TABLE

    import psycopg2

    conn = psycopg2.connect(SUPABASE_DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (schema, table),
            )
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def _require_cache() -> SemanticCacheWrapper:
    if CACHE is None:
        raise RuntimeError("Cache not initialised")
    return CACHE


def _extract_like_values_from_sql(sql: str) -> set:
    """Extract the unique keyword values from LIKE '%val%' patterns in the WHERE clause."""
    if not sql:
        return set()
    s = " ".join(sql.replace("\n", " ").split())
    m = re.search(r"\bwhere\b(.*?)(\border\s+by\b|\bgroup\s+by\b|\blimit\b|$)", s, flags=re.IGNORECASE)
    if not m:
        return set()
    where = m.group(1)
    values = set()
    for val in re.findall(r"\blike\b\s+'%([^%']{1,100})%'", where, flags=re.IGNORECASE):
        v = val.strip().lower()
        if v:
            values.add(v)
    return values


def _extract_sql_intent(sql: str) -> str:
    """
    Determine the INTENT (what kind of information is requested) from cached SQL.
    Returns one of: 'count', 'aggregate', 'list'.

    - 'count'     : SELECT COUNT(...) as the main projection
    - 'aggregate' : uses SUM/AVG/MIN/MAX, or GROUP BY with an aggregate function
    - 'list'      : plain column projection (default)

    This is a regex/keyword check — no LLM call.
    """
    if not sql:
        return "list"
    s = " ".join(sql.replace("\n", " ").split()).lower()

    # Look at the projection between SELECT and FROM
    m = re.search(r"\bselect\b(.*?)\bfrom\b", s, flags=re.IGNORECASE)
    projection = m.group(1) if m else s

    # GROUP BY with any aggregate → aggregate breakdown (check before plain count)
    if "group by" in s and re.search(r"\b(count|sum|avg|min|max)\s*\(", s):
        return "aggregate"

    # COUNT in the projection (no group by) → single counting intent
    if re.search(r"\bcount\s*\(", projection):
        return "count"

    # Other aggregates in the projection → aggregate intent
    if re.search(r"\b(sum|avg|average|min|max|median)\s*\(", projection):
        return "aggregate"

    # Default: listing rows/columns
    return "list"


def _extract_group_by_columns(sql: str) -> list:
    """
    Extract the GROUP BY columns (the aggregation axis) from SQL.
    Returns a sorted list of normalized column names, or [] if no GROUP BY.

    Examples:
    - '... GROUP BY country'                  -> ['country']
    - '... GROUP BY primary_industry, year'   -> ['primary_industry', 'year']
    - '... GROUP BY ALL'                       -> ['__all__']  (DuckDB shorthand)
    - no GROUP BY                              -> []

    Strips table prefixes (cb.country -> country) and ignores ASC/DESC.
    Regex only, no LLM.
    """
    if not sql:
        return []
    s = " ".join(sql.replace("\n", " ").split())
    m = re.search(
        r"\bgroup\s+by\b(.*?)(\border\s+by\b|\bhaving\b|\blimit\b|\bwindow\b|$)",
        s, flags=re.IGNORECASE
    )
    if not m:
        return []
    clause = m.group(1).strip()
    if not clause:
        return []

    # DuckDB 'GROUP BY ALL' — treat as a single sentinel so two ALL queries match
    if clause.lower().strip() == "all":
        return ["__all__"]

    cols = []
    for raw in clause.split(","):
        part = raw.strip().lower()
        # Drop trailing ASC/DESC and surrounding quotes/parens
        part = re.sub(r"\s+(asc|desc)$", "", part).strip(" \"'`()")
        # Strip table alias prefix: cb.country -> country
        if "." in part:
            part = part.split(".")[-1]
        if part:
            cols.append(part)
    return sorted(set(cols))


def _llm_check_filter_match(question: str, sql_filter_values: set, sql_intent: str = "list",
                            sql_group_by: list = None) -> dict:
    """
    Check that a question matches the cached SQL on THREE independent gates.
    All gates must pass for a cache hit:

    1. FILTER VALUE — the domain/category/value asked about must match the
       cached SQL's WHERE filters (healthcare == healthcare, medical != healthcare).
    2. INTENT — list / count / aggregate must match
       ('list companies' != 'how many companies').
    3. GROUP BY AXIS — if the cached SQL groups by some column(s), the question
       must ask for the same breakdown ('by country' != 'by industry').

    Args:
        question: incoming user question
        sql_filter_values: filter values from cached SQL WHERE clause
        sql_intent: 'list' | 'count' | 'aggregate'
        sql_group_by: list of GROUP BY columns from cached SQL ([] if none)

    Returns: {"match": bool, "extracted_value", "extracted_intent",
              "extracted_group_by", "method"}
    """
    if sql_group_by is None:
        sql_group_by = []
    q_lower = question.lower()

    # --- Step A: determine the QUESTION's intent via fast keyword check ---
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b|\bhow much\b", q_lower):
        q_intent = "count"
    elif re.search(r"\b(total|sum|average|avg|maximum|minimum|highest|lowest|mean|median)\b", q_lower):
        q_intent = "aggregate"
    else:
        q_intent = "list"

    # Intent gate (fast). If the question groups by something, it's a breakdown,
    # which behaves like an aggregate even if phrased as "how many ... by ...".
    q_has_breakdown = bool(re.search(r"\b(per|by|grouped by|for each|broken down by)\b", q_lower))

    if q_intent != sql_intent:
        # Allow one nuance: cached aggregate breakdown vs "how many ... per ..."
        if not (sql_intent == "aggregate" and q_intent == "count" and q_has_breakdown):
            return {
                "match": False, "extracted_value": None, "extracted_intent": q_intent,
                "extracted_group_by": None,
                "method": f"intent_mismatch:sql={sql_intent},question={q_intent}",
            }

    # --- Step B: GROUP BY axis gate ---
    # If cached SQL has no GROUP BY but the question clearly asks for a breakdown
    # ("... by country"), they don't match.
    if not sql_group_by and q_has_breakdown:
        return {
            "match": False, "extracted_value": None, "extracted_intent": q_intent,
            "extracted_group_by": "<some>",
            "method": "group_by_mismatch:sql=none,question=breakdown",
        }

    # If cached SQL HAS a GROUP BY, we must verify the question asks for the SAME axis.
    # Natural language axis ("by country") needs the LLM to map to a column, so we
    # fall through to the LLM step below whenever sql_group_by is non-empty.
    needs_llm_for_group_by = bool(sql_group_by)

    # --- Step C: filter value fast path (only safe when no GROUP BY to verify) ---
    if not needs_llm_for_group_by:
        q_words = set(re.findall(r'[a-zA-Z0-9]+', q_lower))
        for val in sql_filter_values:
            if val in q_words:
                return {"match": True, "extracted_value": val, "extracted_intent": q_intent,
                        "extracted_group_by": [], "method": "whole_word_match"}
        for word in q_words:
            for val in sql_filter_values:
                if len(val) >= 4 and val in word:
                    return {"match": True, "extracted_value": word, "extracted_intent": q_intent,
                            "extracted_group_by": [], "method": "word_contains_filter"}

    # --- Step D: LLM extraction (filter value + intent + group-by axis) ---
    if not OPENAI_API_KEY:
        return {"match": False, "extracted_value": None, "extracted_intent": q_intent,
                "extracted_group_by": None, "method": "no_api_key"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        resp = client.chat.completions.create(
            model=SIGNATURE_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze a database question and extract three things:\n"
                        "1. \"value\": the specific domain/category/filter value asked about (lowercase keyword)\n"
                        "2. \"intent\": 'list' (wants rows/names), 'count' (wants a number / how many), "
                        "or 'aggregate' (wants sum/avg/min/max/total)\n"
                        "3. \"group_by\": a list of the column(s) the user wants results broken down BY "
                        "(e.g. 'by country' -> [\"country\"], 'per industry' -> [\"industry\"], "
                        "'for each year' -> [\"year\"]). Use [] if the user wants no breakdown.\n\n"
                        "Return ONLY JSON with fields \"value\", \"intent\", \"group_by\".\n"
                        "Examples:\n"
                        "- 'list healthcare startups' -> {\"value\":\"healthcare\",\"intent\":\"list\",\"group_by\":[]}\n"
                        "- 'how many IoT companies' -> {\"value\":\"iot\",\"intent\":\"count\",\"group_by\":[]}\n"
                        "- 'number of companies by country' -> {\"value\":null,\"intent\":\"count\",\"group_by\":[\"country\"]}\n"
                        "- 'count of startups per industry' -> {\"value\":null,\"intent\":\"count\",\"group_by\":[\"industry\"]}\n"
                        "- 'total funding by region' -> {\"value\":null,\"intent\":\"aggregate\",\"group_by\":[\"region\"]}\n"
                        "If no filter value, use null. If no breakdown, use []."
                    )
                },
                {"role": "user", "content": f"Question: {question}\n\nExtract value, intent, group_by."}
            ],
            response_format={"type": "json_object"},
            timeout=10,
        )

        content = resp.choices[0].message.content
        parsed = json.loads(content)
        extracted = parsed.get("value")
        extracted_intent = (parsed.get("intent") or "list").strip().lower()
        extracted_group_by = parsed.get("group_by") or []
        # Normalize the question's group-by columns the same way as the SQL's
        norm_q_group_by = sorted({
            str(g).strip().lower().split(".")[-1] for g in extracted_group_by if str(g).strip()
        })

        # Gate 2 (intent) — re-check with the LLM's reading
        if extracted_intent != sql_intent:
            if not (sql_intent == "aggregate" and extracted_intent == "count" and norm_q_group_by):
                return {
                    "match": False,
                    "extracted_value": (str(extracted).strip().lower() if extracted else None),
                    "extracted_intent": extracted_intent, "extracted_group_by": norm_q_group_by,
                    "method": f"llm_intent_mismatch:sql={sql_intent},question={extracted_intent}",
                }

        # Gate 3 (group-by axis) — must match the cached SQL's axis
        sql_gb_norm = sorted(c for c in sql_group_by if c != "__all__")
        if sql_gb_norm or norm_q_group_by:
            # Compare as sets; if either side names an axis, they must agree.
            # Allow loose match: question axis is contained in / contains SQL axis
            if not _group_by_axes_match(sql_gb_norm, norm_q_group_by):
                return {
                    "match": False,
                    "extracted_value": (str(extracted).strip().lower() if extracted else None),
                    "extracted_intent": extracted_intent, "extracted_group_by": norm_q_group_by,
                    "method": f"group_by_mismatch:sql={sql_gb_norm},question={norm_q_group_by}",
                }

        # Gate 1 (filter value)
        if not extracted:
            return {"match": True, "extracted_value": None, "extracted_intent": extracted_intent,
                    "extracted_group_by": norm_q_group_by, "method": "no_filter_intent"}

        extracted_lower = str(extracted).strip().lower()
        for val in sql_filter_values:
            if val == extracted_lower:
                return {"match": True, "extracted_value": extracted_lower, "extracted_intent": extracted_intent,
                        "extracted_group_by": norm_q_group_by, "method": "llm_exact_match"}
            if val in extracted_lower or extracted_lower in val:
                return {"match": True, "extracted_value": extracted_lower, "extracted_intent": extracted_intent,
                        "extracted_group_by": norm_q_group_by, "method": "llm_substring_match"}

        # If SQL had no filter values at all but axis+intent matched, accept.
        if not sql_filter_values:
            return {"match": True, "extracted_value": extracted_lower, "extracted_intent": extracted_intent,
                    "extracted_group_by": norm_q_group_by, "method": "axis_intent_match_no_filter"}

        return {"match": False, "extracted_value": extracted_lower, "extracted_intent": extracted_intent,
                "extracted_group_by": norm_q_group_by, "method": "llm_value_mismatch"}

    except Exception as e:
        print(f"_llm_check_filter_match error: {e}")
        return {"match": False, "extracted_value": None, "extracted_intent": q_intent,
                "extracted_group_by": None, "method": f"llm_error:{e}"}


def _group_by_axes_match(sql_axis: list, question_axis: list) -> bool:
    """
    Decide whether two GROUP BY axes refer to the same breakdown.
    Both are normalized lowercase column-name lists.

    - Both empty            -> match (no breakdown on either side)
    - One empty, other not   -> mismatch (one wants a breakdown, the other doesn't)
    - Both non-empty         -> match if they share the same core column name,
      allowing synonyms via substring (e.g. 'country' vs 'country_name',
      'industry' vs 'primary_industry').
    """
    if not sql_axis and not question_axis:
        return True
    if bool(sql_axis) != bool(question_axis):
        return False
    # Both non-empty: require every question axis col to correspond to a SQL axis col
    for q in question_axis:
        matched = False
        for s in sql_axis:
            if q == s or q in s or s in q:
                matched = True
                break
        if not matched:
            return False
    return True


def _lookup_cached_sql(question: str) -> Optional[str]:
    """
    Run the full cache lookup pipeline (semantic → reranker → signature).
    Returns the cached SQL string on a confirmed hit, or None on miss/rejection.
    Used by /query/generate.

    Signature check has two paths:
    - SQL uses equality/IN filters → GPT-based signature comparison (existing logic)
    - SQL uses only LIKE filters   → fast text check: question must contain at least
                                     one of the LIKE keyword values
    - SQL has no filters           → trust semantic + reranker, allow hit
    """
    if CACHE is None:
        return None
    try:
        try:
            raw_candidates = CACHE.cache.check(
                question, distance_threshold=SEMANTIC_DISTANCE_THRESHOLD, num_results=10
            )
        except TypeError:
            raw_candidates = CACHE.cache.check(
                query=question, distance_threshold=SEMANTIC_DISTANCE_THRESHOLD, num_results=10
            )
        if not raw_candidates:
            return None

        chosen = raw_candidates[0]

        # Reranker stage
        if ENABLE_RERANKER and RERANKER is not None:
            reranked = rerank_candidates(question, raw_candidates)
            if not reranked:
                return None
            chosen = reranked[0]
            reranker_distance = chosen.get("reranker_distance")
            if reranker_distance is not None and reranker_distance > RERANKER_DISTANCE_THRESHOLD:
                return None

        # Filter-signature stage
        if ENABLE_SIGNATURE:
            cached_prompt = chosen.get("prompt", "")
            cached_sql = chosen.get("response", "")

            chosen_distance = float(chosen.get("vector_distance", 1.0))
            equality_filters = _extract_filters_from_sql(cached_sql)

            if equality_filters:
                # Existing GPT-based path (SQL uses col = 'val' or col IN (...))
                meta = load_meta(cached_prompt)
                cached_f_hash = (meta or {}).get("filter_signature_hash")
                if not cached_f_hash:
                    sql_sig = canonicalize_filter_sig(
                        {"table": SIGNATURE_TABLE, "filters": equality_filters}
                    )
                    if sql_sig.get("filters"):
                        cached_f_hash = filter_signature_hash(sql_sig)
                        store_filter_signature_hash(cached_prompt, cached_f_hash, TTL_SECONDS)
                if not cached_f_hash:
                    return None
                columns_from_sql = [f["column"] for f in equality_filters]
                q_sig = build_filter_signature(question, columns_from_sql)
                if not q_sig.get("filters"):
                    return None
                if filter_signature_hash(q_sig) != cached_f_hash:
                    return None
            else:
                # No equality/IN filters — check for LIKE patterns.
                # Skip the keyword text check when semantic distance is near-zero:
                # at that point the question is essentially the same cached one
                # (handles typos like "eduaction" vs "education").
                like_values = _extract_like_values_from_sql(cached_sql)
                if like_values and chosen_distance >= 0.05:
                    q_lower = question.lower()
                    if not any(v in q_lower for v in like_values):
                        print(f"_lookup: LIKE keyword mismatch — sql has {like_values}, "
                              f"question='{question[:60]}'")
                        return None
                # chosen_distance < 0.05  → near-exact match, skip text check
                # like_values empty       → no filters, trust semantic + reranker

            return cached_sql

        return chosen.get("response")
    except Exception as e:
        print(f"_lookup_cached_sql error (non-fatal): {e}")
        return None


# ---------------------------------------------------------------------------
# MotherDuck schema loader
# ---------------------------------------------------------------------------

def load_schema_from_motherduck() -> str:
    """
    Connect to MotherDuck, query duckdb_tables() and duckdb_columns() for the
    configured database, and return a human-readable schema context string.
    Called once at startup; result is stored in SCHEMA_CONTEXT.
    """
    import duckdb

    conn_str = f"md:{MOTHERDUCK_DATABASE}?motherduck_token={MOTHERDUCK_TOKEN}"
    conn = duckdb.connect(conn_str)
    try:
        tables = conn.execute(
            "SELECT table_name, comment FROM duckdb_tables() "
            "WHERE database_name = ? AND schema_name = 'main' ORDER BY table_name",
            [MOTHERDUCK_DATABASE],
        ).fetchall()

        cols_raw = conn.execute(
            "SELECT table_name, column_name, data_type, comment FROM duckdb_columns() "
            "WHERE database_name = ? AND schema_name = 'main' ORDER BY table_name, column_index",
            [MOTHERDUCK_DATABASE],
        ).fetchall()
    finally:
        conn.close()

    # Index columns by table
    cols_by_table: Dict[str, List] = {}
    for table_name, col_name, dtype, cmt in cols_raw:
        cols_by_table.setdefault(table_name, []).append((col_name, dtype, cmt or ""))

    lines = [f"Database: {MOTHERDUCK_DATABASE}\n"]
    for table_name, table_comment in tables:
        lines.append(f"Table: {table_name}")
        if table_comment:
            lines.append(f"  Comment: {table_comment}")
        lines.append("  Columns:")
        for col_name, dtype, cmt in cols_by_table.get(table_name, []):
            col_line = f"    - {col_name} ({dtype})"
            if cmt:
                col_line += f": {cmt}"
            lines.append(col_line)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MotherDuck query executor
# ---------------------------------------------------------------------------

def execute_on_motherduck(sql: str) -> List[Dict]:
    """Execute a read-only DuckDB SQL query against MotherDuck and return rows as dicts."""
    import duckdb

    conn = duckdb.connect(f"md:{MOTHERDUCK_DATABASE}?motherduck_token={MOTHERDUCK_TOKEN}")
    try:
        df = conn.execute(sql).fetchdf()
        # Convert any non-serialisable types (e.g. numpy int64) to plain Python
        return json.loads(df.to_json(orient="records", date_format="iso"))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Claude SQL generator
# ---------------------------------------------------------------------------

def _extract_sql_from_text(text: str) -> str:
    """Strip markdown code fences from a Claude response, returning bare SQL."""
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def generate_sql_with_claude(
    question: str,
    error_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Call Claude to generate DuckDB SQL from a natural-language question.

    The schema context is sent with cache_control so repeated calls within the
    5-minute TTL reuse the prompt cache instead of re-tokenising the full schema.
    Returns the raw SQL string (no markdown fences).
    """
    import anthropic

    if not SCHEMA_CONTEXT:
        raise RuntimeError("SCHEMA_CONTEXT is empty — schema was not loaded at startup.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # System: static instructions (no cache_control needed — tiny)
    # + large schema block with cache_control (cached for 5 min)
    system_blocks = [
        {
            "type": "text",
            "text": (
                "You are an expert SQL analyst specialised in DuckDB. "
                f"Generate valid DuckDB-compatible SQL for the `{MOTHERDUCK_DATABASE}` database. "
                "Rules: use only tables and columns that exist in the schema below; "
                "qualify table names as needed; "
                "return ONLY the raw SQL — no markdown, no explanation, no code fences."
            ),
        },
        {
            "type": "text",
            "text": f"Schema:\n\n{SCHEMA_CONTEXT}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    # Build user turn
    user_parts = [f"Question: {question}"]
    if error_history:
        user_parts.append("\nPrevious failed attempts (fix the errors):")
        for i, err in enumerate(error_history, 1):
            user_parts.append(f"\nAttempt {i}:\nSQL:\n{err['sql']}\nError: {err['error']}")

    response = client.messages.create(
        model=SQL_AGENT_MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system_blocks,
        messages=[{"role": "user", "content": "\n".join(user_parts)}],
    )

    # Extract text block (thinking blocks appear first but we want the text)
    for block in response.content:
        if block.type == "text":
            return _extract_sql_from_text(block.text)

    raise RuntimeError("Claude returned no text block in its response.")


# ---------------------------------------------------------------------------
# Startup initialization
# Same as beam_cache_lookup.py::on_start() + beam_cache_store.py::on_start()
# ---------------------------------------------------------------------------

def on_start():
    """Called once when the app starts. Same as Beam on_start()."""
    global CACHE, REDIS_CLIENT, TABLE_COLUMNS, RERANKER, SCHEMA_CONTEXT

    print("DEBUG: on_start() - Initializing components...")

    # 1. Redis client for meta storage
    REDIS_CLIENT = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    print("DEBUG: Redis client initialized")

    # 2. SemanticCacheWrapper - SAME as Beam on_start()
    cfg: Dict[str, Any] = {
        "redis_url": REDIS_URL,
        "cache_name": CACHE_NAME,
        "distance_threshold": SEMANTIC_DISTANCE_THRESHOLD,
        "ttl_seconds": TTL_SECONDS,
    }
    CACHE = SemanticCacheWrapper.from_config(cfg)
    print(f"DEBUG: SemanticCacheWrapper initialized: {CACHE_NAME}")

    # 3. Try to create/ensure index (same as Beam on_start)
    try:
        if hasattr(CACHE, "create_index") and callable(getattr(CACHE, "create_index")):
            CACHE.create_index(overwrite=False)
        elif hasattr(CACHE, "ensure_index") and callable(getattr(CACHE, "ensure_index")):
            CACHE.ensure_index()
        elif hasattr(CACHE, "index") and hasattr(CACHE.index, "create"):
            CACHE.index.create(overwrite=False)
    except Exception:
        pass

    # 4. Reranker - SAME as Beam on_start()
    if ENABLE_RERANKER:
        print(f"DEBUG: Initializing reranker: {RERANKER_MODEL}")
        ce = CrossEncoder(model_name_or_path=RERANKER_MODEL)
        RERANKER = ce.create_reranker()
        print("DEBUG: Reranker initialized")

    # 5. Table columns derived from cached SQL at lookup time — no Supabase needed

    # 6. Load MotherDuck schema for SQL generation (only when token is configured)
    if MOTHERDUCK_TOKEN:
        try:
            print(f"DEBUG: Loading schema from MotherDuck ({MOTHERDUCK_DATABASE})...")
            SCHEMA_CONTEXT = load_schema_from_motherduck()
            table_count = SCHEMA_CONTEXT.count("\nTable:")
            print(f"DEBUG: Schema loaded — {table_count} tables")
        except Exception as e:
            print(f"WARNING: Could not load MotherDuck schema: {e}")
            print("WARNING: /query/generate will be unavailable until schema is loaded.")
    else:
        print("DEBUG: MOTHERDUCK_TOKEN not set — skipping schema load (/query/generate disabled)")

    print("DEBUG: on_start() complete")


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Initialize on module load (Lambda cold start)
on_start()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "cache_name": CACHE_NAME,
        "semantic_distance_threshold": SEMANTIC_DISTANCE_THRESHOLD,
        "enable_signature": ENABLE_SIGNATURE,
        "signature_table": SIGNATURE_TABLE,
        "columns_loaded": len(TABLE_COLUMNS) if TABLE_COLUMNS else 0,
        "enable_reranker": ENABLE_RERANKER,
        "reranker_model": RERANKER_MODEL if ENABLE_RERANKER else None,
        "reranker_distance_threshold": RERANKER_DISTANCE_THRESHOLD if ENABLE_RERANKER else None,
        "sql_agent": {
            "model": SQL_AGENT_MODEL,
            "max_retries": SQL_AGENT_MAX_RETRIES,
            "database": MOTHERDUCK_DATABASE,
            "schema_loaded": bool(SCHEMA_CONTEXT),
            "schema_tables": SCHEMA_CONTEXT.count("\nTable:") if SCHEMA_CONTEXT else 0,
        },
    }), 200


@app.route("/cache/store", methods=["POST"])
def store_cache_endpoint():
    """
    Store a question/SQL pair in the semantic cache.
    Same logic as beam_cache_store.py::store_cache()
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        question = data.get("question")
        sql = data.get("sql")
        ttl_seconds_input = data.get("ttl_seconds")

        if not question or not sql:
            return jsonify({"success": False, "error": "Both 'question' and 'sql' are required."}), 400

        c = _require_cache()
        ttl_seconds = int(ttl_seconds_input or TTL_SECONDS)
        sig_hash = None

        # Store into semantic cache (same as Beam)
        _cache_store(c, question, sql, ttl_seconds)
        print(f"STORE: Stored question='{question[:60]}...' ttl={ttl_seconds}")

        # Store signature hash (filters only)
        if ENABLE_SIGNATURE:
            sql_sig = {"table": SIGNATURE_TABLE, "filters": _extract_filters_from_sql(sql)}
            sql_sig = canonicalize_filter_sig(sql_sig)
            if sql_sig.get("filters"):
                sig_hash = filter_signature_hash(sql_sig)
                store_filter_signature_hash(question, sig_hash, ttl_seconds)
                print(f"STORE: Stored signature hash={sig_hash[:16]}...")

        return jsonify({
            "success": True,
            "stored": True,
            "question": question,
            "sql": sql,
            "ttl_seconds": ttl_seconds,
            "signature_hash": sig_hash,
        }), 200

    except Exception as e:
        print(f"STORE ERROR: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "stored": False,
            "error": f"Failed to store in cache: {e}",
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/cache/lookup", methods=["POST"])
def check_cache_endpoint():
    """
    Look up a question in the semantic cache.
    Same logic as beam_cache_lookup.py::check_cache()
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "found": False, "error": "No JSON data provided"}), 400

        question = data.get("question")
        semantic_distance_threshold = data.get("semantic_distance_threshold")

        if not question:
            return jsonify({"success": False, "found": False, "error": "'question' is required"}), 400

        c = _require_cache()
        sem_thr = semantic_distance_threshold if semantic_distance_threshold is not None else SEMANTIC_DISTANCE_THRESHOLD

        print(f"LOOKUP: question='{question[:60]}...' threshold={sem_thr}")

        # 1) Get semantic candidates from cache (same as Beam)
        try:
            raw_candidates = c.cache.check(question, distance_threshold=sem_thr, num_results=10)
        except TypeError:
            raw_candidates = c.cache.check(query=question, distance_threshold=sem_thr, num_results=10)

        print(f"LOOKUP: Found {len(raw_candidates) if raw_candidates else 0} raw candidates")

        if not raw_candidates:
            return jsonify({"success": True, "found": False, "question": question}), 200

        # Log candidates
        for i, cand in enumerate(raw_candidates[:3]):
            print(f"  Candidate {i}: prompt='{str(cand.get('prompt',''))[:50]}...' "
                  f"distance={cand.get('vector_distance', 'N/A')}")

        # Calculate cosine similarity for the top match before reranking
        top_before_rerank = raw_candidates[0]
        vector_distance = float(top_before_rerank.get("vector_distance", 0.0))
        cosine_sim = float((2 - vector_distance) / 2)

        chosen = top_before_rerank
        reranker_distance: Optional[float] = None

        # 2) Optional reranker to pick best candidate (same as Beam)
        if ENABLE_RERANKER and RERANKER is not None:
            try:
                reranked = rerank_candidates(question, raw_candidates)

                if reranked:
                    chosen = reranked[0]
                    reranker_distance = chosen.get("reranker_distance")

                    vector_distance = float(chosen.get("vector_distance", 0.0))
                    cosine_sim = float((2 - vector_distance) / 2)

                    print(f"LOOKUP: Reranker best: distance={reranker_distance}, "
                          f"prompt='{str(chosen.get('prompt',''))[:50]}...'")

                    if reranker_distance is not None and reranker_distance > RERANKER_DISTANCE_THRESHOLD:
                        return jsonify({
                            "success": True,
                            "found": False,
                            "question": question,
                            "cached_question": chosen.get("prompt"),
                            "sql": chosen.get("response"),
                            "vector_distance": vector_distance,
                            "cosine_similarity": cosine_sim,
                            "reranker_distance": reranker_distance,
                            "rejected_reason": f"reranker_distance>{RERANKER_DISTANCE_THRESHOLD}",
                        }), 200
            except Exception as e:
                print(f"LOOKUP: Reranker error: {e}")
                return jsonify({
                    "success": True,
                    "found": False,
                    "question": question,
                    "cached_question": top_before_rerank.get("prompt"),
                    "sql": top_before_rerank.get("response"),
                    "vector_distance": vector_distance,
                    "cosine_similarity": cosine_sim,
                    "rejected_reason": f"reranker_error:{e}",
                }), 200

        # 3) Filter-signature gate (WHERE-only) - unified approach
        #    Uses LLM to extract filter values from SQL, then checks if question matches
        if ENABLE_SIGNATURE:
            try:
                cached_prompt = chosen.get("prompt", "")
                cached_sql = chosen.get("response", "")
                chosen_distance = float(chosen.get("vector_distance", 0.0))

                # Near-exact semantic match (distance < 0.05) — trust it, skip filter check
                if chosen_distance < 0.05:
                    print(f"LOOKUP: Near-exact match (dist={chosen_distance:.4f}), skipping filter check")
                else:
                    # Extract ALL filter values from cached SQL (both LIKE and equality)
                    equality_filters = _extract_filters_from_sql(cached_sql)
                    like_values = _extract_like_values_from_sql(cached_sql)

                    # Combine all filter values into one set
                    all_filter_values = set()
                    for f in equality_filters:
                        val = f.get("value", "")
                        if isinstance(val, list):
                            all_filter_values.update(v for v in val if v)
                        elif val:
                            all_filter_values.add(val)
                    all_filter_values.update(like_values)

                    # Determine the cached SQL's intent and GROUP BY axis
                    sql_intent = _extract_sql_intent(cached_sql)
                    sql_group_by = _extract_group_by_columns(cached_sql)

                    # Check filter value + intent + group-by axis against the question.
                    filter_match = _llm_check_filter_match(question, all_filter_values, sql_intent, sql_group_by)

                    if not filter_match["match"]:
                        print(f"LOOKUP: Rejected — sql_filters={all_filter_values}, sql_intent={sql_intent}, "
                              f"sql_group_by={sql_group_by}, method={filter_match.get('method')} — REJECTED")
                        return jsonify({
                            "success": True,
                            "found": False,
                            "question": question,
                            "cached_question": cached_prompt,
                            "sql": cached_sql,
                            "vector_distance": vector_distance,
                            "cosine_similarity": cosine_sim,
                            "reranker_distance": reranker_distance,
                            "sql_intent": sql_intent,
                            "question_intent": filter_match.get("extracted_intent"),
                            "sql_group_by": sql_group_by,
                            "question_group_by": filter_match.get("extracted_group_by"),
                            "rejected_reason": f"{filter_match.get('method')}:sql_filters={all_filter_values},question_value={filter_match.get('extracted_value', '?')}",
                        }), 200
                    else:
                        print(f"LOOKUP: Match confirmed — value={filter_match.get('extracted_value')}, "
                              f"intent={sql_intent}, group_by={sql_group_by}, method={filter_match.get('method')}")

                # All checks passed!
                print("LOOKUP: HIT - found matching cached entry")
                return jsonify({
                    "success": True,
                    "found": True,
                    "question": question,
                    "cached_question": cached_prompt,
                    "sql": cached_sql,
                    "vector_distance": vector_distance,
                    "cosine_similarity": cosine_sim,
                    "reranker_distance": reranker_distance,
                    "signature_match": True,
                }), 200

            except Exception as e:
                print(f"LOOKUP: Signature error: {e}")
                return jsonify({
                    "success": True,
                    "found": False,
                    "question": question,
                    "cached_question": chosen.get("prompt"),
                    "sql": chosen.get("response"),
                    "vector_distance": vector_distance,
                    "cosine_similarity": cosine_sim,
                    "reranker_distance": reranker_distance,
                    "rejected_reason": f"signature_error:{e}",
                }), 200

        # Signatures disabled => accept after semantic (+ optional rerank)
        print("LOOKUP: HIT (no signature check)")
        return jsonify({
            "success": True,
            "found": True,
            "question": question,
            "cached_question": chosen.get("prompt"),
            "sql": chosen.get("response"),
            "vector_distance": vector_distance,
            "cosine_similarity": cosine_sim,
            "reranker_distance": reranker_distance,
        }), 200

    except Exception as e:
        print(f"LOOKUP ERROR: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "found": False,
            "question": data.get("question") if data else None,
            "error": f"Failed to check cache: {e}",
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/query/generate", methods=["POST"])
def query_generate():
    """
    Generate SQL from a natural-language question, execute it against MotherDuck,
    and auto-store the successful (question, SQL) pair in the semantic cache.

    Request body:  {"question": "Show me top funded startups"}
    Response:      {success, from_cache, question, sql, data, row_count, attempts}

    Typical workflow:
      1. Call POST /cache/lookup first.
      2. If found=true → use cached SQL directly (or call this endpoint to also execute it).
      3. If found=false → call this endpoint to generate, execute, and cache.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"success": False, "error": "'question' is required"}), 400

        # Guard: prerequisites
        if not MOTHERDUCK_TOKEN:
            return jsonify({"success": False, "error": "MOTHERDUCK_TOKEN is not configured"}), 500
        if not ANTHROPIC_API_KEY:
            return jsonify({"success": False, "error": "ANTHROPIC_API_KEY is not configured"}), 500
        if not SCHEMA_CONTEXT:
            return jsonify({"success": False, "error": "Schema not loaded (check MOTHERDUCK_TOKEN)"}), 500

        print(f"GENERATE: question='{question[:80]}'")

        # Check cache first — skip Claude entirely on a confirmed hit
        cached_sql = _lookup_cached_sql(question)
        if cached_sql:
            print("GENERATE: Cache hit — executing cached SQL")
            results = execute_on_motherduck(cached_sql)
            return jsonify({
                "success": True,
                "from_cache": True,
                "question": question,
                "sql": cached_sql,
                "row_count": len(results),
                "data": results,
            }), 200

        error_history: List[Dict[str, str]] = []
        last_sql: Optional[str] = None
        last_error: Optional[str] = None

        for attempt in range(1, SQL_AGENT_MAX_RETRIES + 1):
            print(f"GENERATE: Attempt {attempt}/{SQL_AGENT_MAX_RETRIES}")
            try:
                sql = generate_sql_with_claude(question, error_history or None)
                last_sql = sql
                print(f"GENERATE: SQL generated: {sql[:120]}")

                results = execute_on_motherduck(sql)
                print(f"GENERATE: Executed OK — {len(results)} rows returned")

                # Auto-store in semantic cache
                try:
                    c = _require_cache()
                    _cache_store(c, question, sql, TTL_SECONDS)
                    if ENABLE_SIGNATURE:
                        sql_sig = canonicalize_filter_sig(
                            {"table": SIGNATURE_TABLE, "filters": _extract_filters_from_sql(sql)}
                        )
                        if sql_sig.get("filters"):
                            sig_hash = filter_signature_hash(sql_sig)
                            store_filter_signature_hash(question, sig_hash, TTL_SECONDS)
                    print("GENERATE: Stored in cache")
                except Exception as ce:
                    print(f"GENERATE: Cache store skipped (non-fatal): {ce}")

                return jsonify({
                    "success": True,
                    "from_cache": False,
                    "question": question,
                    "sql": sql,
                    "attempts": attempt,
                    "row_count": len(results),
                    "data": results,
                }), 200

            except Exception as exec_err:
                last_error = str(exec_err)
                error_history.append({"sql": last_sql or "", "error": last_error})
                print(f"GENERATE: Attempt {attempt} failed: {last_error}")

        # All retries exhausted
        return jsonify({
            "success": False,
            "question": question,
            "sql": last_sql,
            "error": f"Failed after {SQL_AGENT_MAX_RETRIES} attempt(s). Last error: {last_error}",
            "error_history": error_history,
        }), 500

    except Exception as e:
        print(f"GENERATE ERROR: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


@app.route("/debug/candidates", methods=["POST"])
def debug_candidates():
    """Debug endpoint: inspect raw semantic matches with NO threshold filtering."""
    try:
        data = request.get_json()
        question = data.get("question", "") if data else ""
        if not question:
            return jsonify({"error": "'question' is required"}), 400

        c = _require_cache()

        # Use max threshold to return ALL entries
        try:
            raw = c.cache.check(question, num_results=20, distance_threshold=2.0)
        except Exception as e:
            return jsonify({"error": f"Cache check failed: {e}"}), 500

        out = []
        for m in raw or []:
            vd = float(m.get("vector_distance", 0.0))
            out.append({
                "cached_question": m.get("prompt"),
                "cached_sql": m.get("response"),
                "vector_distance": vd,
                "cosine_similarity": float((2 - vd) / 2),
            })

        return jsonify({
            "question": question,
            "configured_threshold": SEMANTIC_DISTANCE_THRESHOLD,
            "count": len(out),
            "candidates": out,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------
# Uncomment for Lambda deployment:
# from apig_wsgi import make_lambda_handler
# lambda_handler = make_lambda_handler(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8001)
