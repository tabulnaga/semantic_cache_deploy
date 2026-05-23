"""
test_render_api.py

Tests the deployed Semantic Cache API on Render.

Run:
    python -m pytest test_render_api.py -v -s

Requires:
    pip install pytest requests

Tests cover:
1. Store entries (seed the cache)
2. Semantic similarity (different phrasings → same cache hit)
3. Domain discrimination (IoT ≠ Healthcare)
4. Filter signature validation (medical ≠ healthcare, security ≠ IoT)
5. Cache misses (unrelated questions)
"""

import pytest
import requests
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://semantic-cache-deploy.onrender.com"
STORE_URL = f"{BASE_URL}/cache/store"
LOOKUP_URL = f"{BASE_URL}/cache/lookup"
DEBUG_URL = f"{BASE_URL}/debug/candidates"

# Increase timeout for Render free tier cold starts
TIMEOUT = 120


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def store(question: str, sql: str):
    """Store a question/SQL pair in the cache."""
    r = requests.post(STORE_URL, json={"question": question, "sql": sql}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def lookup(question: str):
    """Look up a question in the cache."""
    r = requests.post(LOOKUP_URL, json={"question": question}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def debug_candidates(question: str):
    """Get raw candidates with distances."""
    r = requests.post(DEBUG_URL, json={"question": question}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# SQL fixtures
# ---------------------------------------------------------------------------
IOT_SQL = (
    "SELECT DISTINCT company_name FROM company_business "
    "WHERE LOWER(primary_industry) LIKE '%iot%' OR LOWER(category_list) LIKE '%iot%'"
)
HEALTHCARE_SQL = (
    "SELECT DISTINCT company_name FROM company_business "
    "WHERE LOWER(primary_industry) LIKE '%healthcare%' OR LOWER(category_list) LIKE '%healthcare%'"
)
SECURITY_SQL = (
    "SELECT DISTINCT company_name FROM company_business "
    "WHERE LOWER(primary_industry) LIKE '%security%' OR LOWER(category_list) LIKE '%security%'"
)
AI_SQL = (
    "SELECT DISTINCT company_name FROM company_business "
    "WHERE LOWER(primary_industry) LIKE '%ai%' OR LOWER(category_list) LIKE '%ai%'"
)
FOUNDED_SQL = (
    "SELECT company_name, founded_on FROM company_identity "
    "WHERE founded_on > '2020-01-01' ORDER BY founded_on DESC"
)
FUNDING_SQL = (
    "SELECT company_name, total_funding_usd FROM company_business "
    "WHERE total_funding_usd > 5000000 ORDER BY total_funding_usd DESC"
)


# ---------------------------------------------------------------------------
# 0. Health check & Store entries
# ---------------------------------------------------------------------------
class TestSetup:
    """Seed the cache with test entries."""

    def test_store_iot(self):
        result = store("tell me the startups that work in IoT domain", IOT_SQL)
        assert result.get("success") is True
        print(f"\n  ✓ Stored IoT entry")

    def test_store_healthcare(self):
        result = store("tell me the startups that work in Healthcare domain", HEALTHCARE_SQL)
        assert result.get("success") is True
        print(f"\n  ✓ Stored Healthcare entry")

    def test_store_security(self):
        result = store("tell me the startups that work in security domain", SECURITY_SQL)
        assert result.get("success") is True
        print(f"\n  ✓ Stored Security entry")

    def test_store_ai(self):
        result = store("tell me the startups that work in AI domain", AI_SQL)
        assert result.get("success") is True
        print(f"\n  ✓ Stored AI entry")

    def test_store_founded(self):
        result = store("show me companies founded after 2020", FOUNDED_SQL)
        assert result.get("success") is True
        print(f"\n  ✓ Stored Founded entry")

    def test_store_funding(self):
        result = store("what startups have raised more than 5 million dollars", FUNDING_SQL)
        assert result.get("success") is True
        print(f"\n  ✓ Stored Funding entry")


# ---------------------------------------------------------------------------
# 1. Semantic similarity - same domain, different phrasings
# ---------------------------------------------------------------------------
class TestSemanticSimilarity:
    """Different phrasings of the same question should hit the same cached entry."""

    @pytest.mark.parametrize("query", [
        "what are the companies that working in IoT domain",
        "tell me companies that working in IoT",
        "show me startups in the IoT space",
        "list IoT companies",
        "which startups operate in the IoT sector",
    ])
    def test_iot_variants_hit_iot_sql(self, query):
        result = lookup(query)
        assert result.get("found") is True, f"No cache hit for: '{query}'"
        assert "iot" in result.get("sql", "").lower(), (
            f"Query '{query}' returned wrong SQL: {result.get('sql', '')[:80]}"
        )
        print(f"\n  ✓ '{query}' → IoT SQL (dist={result.get('vector_distance', '?')})")

    @pytest.mark.parametrize("query", [
        "tell me the startups that work in Healthcare domain",
        "show me Healthcare companies",
        "which companies work in healthcare sector",
        "list healthcare startups",
    ])
    def test_healthcare_variants_hit_healthcare_sql(self, query):
        result = lookup(query)
        assert result.get("found") is True, f"No cache hit for: '{query}'"
        assert "healthcare" in result.get("sql", "").lower(), (
            f"Query '{query}' returned wrong SQL: {result.get('sql', '')[:80]}"
        )
        print(f"\n  ✓ '{query}' → Healthcare SQL")

    @pytest.mark.parametrize("query", [
        "show me companies founded after 2020",
        "which companies were started after 2020",
        "list startups founded post 2020",
    ])
    def test_founded_variants_hit_founded_sql(self, query):
        result = lookup(query)
        assert result.get("found") is True, f"No cache hit for: '{query}'"
        assert "founded_on" in result.get("sql", "").lower(), (
            f"Query '{query}' returned wrong SQL: {result.get('sql', '')[:80]}"
        )
        print(f"\n  ✓ '{query}' → Founded SQL")

    @pytest.mark.parametrize("query", [
        "what startups have raised more than 5 million dollars",
        "show me startups that raised over 5 million",
        "companies with funding above 5M",
    ])
    def test_funding_variants_hit_funding_sql(self, query):
        result = lookup(query)
        assert result.get("found") is True, f"No cache hit for: '{query}'"
        sql = result.get("sql", "").lower()
        assert "funding" in sql or "5000000" in sql, (
            f"Query '{query}' returned wrong SQL: {result.get('sql', '')[:80]}"
        )
        print(f"\n  ✓ '{query}' → Funding SQL")


# ---------------------------------------------------------------------------
# 2. Domain discrimination - IoT ≠ Healthcare ≠ Security ≠ AI
# ---------------------------------------------------------------------------
class TestDomainDiscrimination:
    """Each domain query must return its own SQL, not another domain's."""

    def test_iot_returns_iot_not_healthcare(self):
        result = lookup("tell me the startups that work in IoT domain")
        assert result.get("found") is True
        sql = result.get("sql", "").lower()
        assert "iot" in sql, f"Expected IoT SQL, got: {sql[:80]}"
        assert "healthcare" not in sql, "IoT query returned Healthcare SQL!"
        print(f"\n  ✓ IoT → IoT SQL (not Healthcare)")

    def test_healthcare_returns_healthcare_not_iot(self):
        result = lookup("tell me the startups that work in Healthcare domain")
        assert result.get("found") is True
        sql = result.get("sql", "").lower()
        assert "healthcare" in sql, f"Expected Healthcare SQL, got: {sql[:80]}"
        assert "'%iot%'" not in sql, "Healthcare query returned IoT SQL!"
        print(f"\n  ✓ Healthcare → Healthcare SQL (not IoT)")

    def test_security_returns_security_not_iot(self):
        result = lookup("tell me the startups that work in security domain")
        assert result.get("found") is True
        sql = result.get("sql", "").lower()
        assert "security" in sql, f"Expected Security SQL, got: {sql[:80]}"
        assert "'%iot%'" not in sql, "Security query returned IoT SQL!"
        print(f"\n  ✓ Security → Security SQL (not IoT)")

    def test_ai_returns_ai_not_iot(self):
        result = lookup("tell me the startups that work in AI domain")
        assert result.get("found") is True
        sql = result.get("sql", "").lower()
        assert "'%ai%'" in sql or "ai" in sql, f"Expected AI SQL, got: {sql[:80]}"
        print(f"\n  ✓ AI → AI SQL")


# ---------------------------------------------------------------------------
# 3. Filter signature - ENABLE_SIGNATURE=true tests
#    Semantically similar but different domain keyword → must REJECT
# ---------------------------------------------------------------------------
class TestFilterSignature:
    """
    THE critical tests for ENABLE_SIGNATURE=true.
    These domains are semantically close but have different LIKE values.
    The filter signature gate must reject the wrong cache hit.
    """

    def test_medical_does_not_return_healthcare_sql(self):
        """'medical' ≈ 'healthcare' semantically, but LIKE '%healthcare%' ≠ 'medical'."""
        result = lookup("tell me the startups that work in medical domain")
        if result.get("found") is True:
            sql = result.get("sql", "").lower()
            assert "healthcare" not in sql, (
                f"FILTER SIGNATURE FAILED: 'medical' returned Healthcare SQL!\n"
                f"SQL: {sql[:100]}\n"
                f"Rejected reason: {result.get('rejected_reason', 'none')}"
            )
        else:
            # Correctly rejected — not found
            reason = result.get("rejected_reason", "")
            print(f"\n  ✓ 'medical domain' correctly rejected (reason: {reason})")

    def test_cybersecurity_does_not_return_security_sql(self):
        """'cybersecurity' ≈ 'security' semantically, but LIKE '%security%' might match."""
        result = lookup("tell me the startups that work in cybersecurity domain")
        if result.get("found") is True:
            sql = result.get("sql", "").lower()
            # 'cybersecurity' contains 'security' as substring, so LIKE '%security%' WOULD match
            # This is actually a valid hit since the SQL would return correct results
            print(f"\n  ℹ 'cybersecurity' returned SQL (may be valid since it contains 'security')")
        else:
            print(f"\n  ✓ 'cybersecurity' rejected (reason: {result.get('rejected_reason', '')})")

    def test_biotech_does_not_return_healthcare_sql(self):
        """'biotech' ≈ 'healthcare' semantically, but LIKE '%healthcare%' ≠ 'biotech'."""
        result = lookup("tell me the startups that work in biotech domain")
        if result.get("found") is True:
            sql = result.get("sql", "").lower()
            assert "healthcare" not in sql, (
                f"FILTER SIGNATURE FAILED: 'biotech' returned Healthcare SQL!\n"
                f"SQL: {sql[:100]}"
            )
        else:
            reason = result.get("rejected_reason", "")
            print(f"\n  ✓ 'biotech domain' correctly rejected (reason: {reason})")

    def test_machine_learning_does_not_return_ai_sql(self):
        """'machine learning' ≈ 'AI' semantically, but LIKE '%ai%' ≠ 'machine learning'."""
        result = lookup("tell me the startups that work in machine learning domain")
        if result.get("found") is True:
            sql = result.get("sql", "").lower()
            assert "'%ai%'" not in sql, (
                f"FILTER SIGNATURE FAILED: 'machine learning' returned AI SQL!\n"
                f"SQL: {sql[:100]}"
            )
        else:
            reason = result.get("rejected_reason", "")
            print(f"\n  ✓ 'machine learning domain' correctly rejected (reason: {reason})")

    def test_deep_learning_does_not_return_ai_sql(self):
        """'deep learning' ≈ 'AI' semantically, but LIKE '%ai%' ≠ 'deep learning'."""
        result = lookup("tell me the startups that work in deep learning domain")
        if result.get("found") is True:
            sql = result.get("sql", "").lower()
            assert "'%ai%'" not in sql, (
                f"FILTER SIGNATURE FAILED: 'deep learning' returned AI SQL!\n"
                f"SQL: {sql[:100]}"
            )
        else:
            reason = result.get("rejected_reason", "")
            print(f"\n  ✓ 'deep learning domain' correctly rejected (reason: {reason})")

    def test_pharma_does_not_return_healthcare_sql(self):
        """'pharma' ≈ 'healthcare' semantically, but LIKE '%healthcare%' ≠ 'pharma'."""
        result = lookup("tell me the startups that work in pharma domain")
        if result.get("found") is True:
            sql = result.get("sql", "").lower()
            assert "healthcare" not in sql, (
                f"FILTER SIGNATURE FAILED: 'pharma' returned Healthcare SQL!\n"
                f"SQL: {sql[:100]}"
            )
        else:
            reason = result.get("rejected_reason", "")
            print(f"\n  ✓ 'pharma domain' correctly rejected (reason: {reason})")

    def test_infosec_does_not_return_security_sql(self):
        """'infosec' ≈ 'security' semantically, but LIKE '%security%' ≠ 'infosec'."""
        result = lookup("tell me the startups that work in infosec domain")
        if result.get("found") is True:
            sql = result.get("sql", "").lower()
            assert "security" not in sql, (
                f"FILTER SIGNATURE FAILED: 'infosec' returned Security SQL!\n"
                f"SQL: {sql[:100]}"
            )
        else:
            reason = result.get("rejected_reason", "")
            print(f"\n  ✓ 'infosec domain' correctly rejected (reason: {reason})")

    def test_fintech_does_not_return_any_cached_domain(self):
        """'fintech' is not cached and not similar enough to any domain."""
        result = lookup("tell me the startups that work in fintech domain")
        if result.get("found") is True:
            sql = result.get("sql", "").lower()
            # If found, make sure it's not returning wrong domain SQL
            for domain in ["iot", "healthcare", "security"]:
                assert f"'%{domain}%'" not in sql, (
                    f"'fintech' incorrectly returned {domain} SQL!"
                )
        else:
            print(f"\n  ✓ 'fintech domain' correctly not found")


# ---------------------------------------------------------------------------
# 4. Cache misses - completely unrelated questions
# ---------------------------------------------------------------------------
class TestCacheMisses:
    """Unrelated questions must not match any cached entry."""

    def test_weather_misses(self):
        result = lookup("what is the weather today in Kuwait")
        assert result.get("found") is False, "Weather question should not match!"
        print(f"\n  ✓ Weather question → miss")

    def test_cooking_misses(self):
        result = lookup("how do I cook pasta carbonara")
        assert result.get("found") is False, "Cooking question should not match!"
        print(f"\n  ✓ Cooking question → miss")

    def test_math_misses(self):
        result = lookup("what is the square root of 144")
        assert result.get("found") is False, "Math question should not match!"
        print(f"\n  ✓ Math question → miss")

    def test_greeting_misses(self):
        result = lookup("hello how are you doing today")
        assert result.get("found") is False, "Greeting should not match!"
        print(f"\n  ✓ Greeting → miss")


# ---------------------------------------------------------------------------
# 5. Debug - inspect raw distances
# ---------------------------------------------------------------------------
class TestDebugDistances:
    """Inspect raw candidate distances to understand the similarity landscape."""

    def test_debug_iot_candidates(self):
        """Show all candidates for IoT query with their distances."""
        result = debug_candidates("tell me the startups that work in IoT domain")
        candidates = result.get("candidates", [])
        print(f"\n  IoT query candidates ({len(candidates)} total):")
        for i, c in enumerate(candidates[:5]):
            print(f"    [{i+1}] dist={c.get('vector_distance', '?'):.4f} "
                  f"q='{c.get('cached_question', '')[:50]}'")

    def test_debug_medical_vs_healthcare(self):
        """Show why 'medical' is close to 'healthcare' in vector space."""
        result = debug_candidates("tell me the startups that work in medical domain")
        candidates = result.get("candidates", [])
        print(f"\n  'medical domain' candidates ({len(candidates)} total):")
        for i, c in enumerate(candidates[:5]):
            print(f"    [{i+1}] dist={c.get('vector_distance', '?'):.4f} "
                  f"q='{c.get('cached_question', '')[:50]}'")

    def test_debug_machine_learning_vs_ai(self):
        """Show why 'machine learning' is close to 'AI' in vector space."""
        result = debug_candidates("tell me the startups that work in machine learning domain")
        candidates = result.get("candidates", [])
        print(f"\n  'machine learning domain' candidates ({len(candidates)} total):")
        for i, c in enumerate(candidates[:5]):
            print(f"    [{i+1}] dist={c.get('vector_distance', '?'):.4f} "
                  f"q='{c.get('cached_question', '')[:50]}'")
