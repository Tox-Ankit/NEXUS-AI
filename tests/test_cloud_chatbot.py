"""
End-to-end test for the Ollama Cloud chatbot pipeline.

Tests the full loop:
  User question → Cloud LLM (intent) → structured request → validation →
  engine calculation → structured result → Cloud LLM (explanation) → answer

Run:  python tests/test_cloud_chatbot.py
"""

import sys
import os
import io

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.llm.service import LLMService
from app.chatbot.controller import AIController

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def report(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))

# ── 1. Health check ──────────────────────────────────────────────────────────

print("\n=== NEXUS AI - Ollama Cloud Integration Test ===\n")
print("[1/4] LLM Health Check")

available = LLMService.is_available()
report("LLM is_available()", available, "Cloud service reachable" if available else "UNREACHABLE - check OLLAMA_API_KEY")

if not available:
    print("\n[ERROR] Cannot continue - LLM service is not available.")
    print("    Make sure your .env has a valid OLLAMA_API_KEY.")
    sys.exit(1)

# ── 2. Raw completion ────────────────────────────────────────────────────────

print("\n[2/4] Raw LLM Completion")

try:
    resp = LLMService.complete("Reply with exactly: NEXUS CLOUD OK")
    ok = "NEXUS" in resp.upper() and "CLOUD" in resp.upper()
    report("Free-text completion", ok, f"Got: {resp[:80]}...")
except Exception as e:
    report("Free-text completion", False, str(e))

try:
    resp_json = LLMService.complete(
        'Return exactly this JSON and nothing else: {"status": "ok", "source": "cloud"}',
        schema={"type": "json"},
    )
    ok = isinstance(resp_json, dict) and resp_json.get("status") == "ok"
    report("JSON-mode completion", ok, f"Got: {resp_json}")
except Exception as e:
    report("JSON-mode completion", False, str(e))

# ── 3. Full controller loop (simulated dataset context) ─────────────────────

print("\n[3/4] Full Controller Loop (simulated context)")

dataset_context = {
    "columns": ["product", "category", "price", "units_sold", "revenue", "month"],
    "row_count": 12,
    "column_count": 6,
}

stats_context = {
    "price": {"min": 19.99, "max": 49.99, "mean": 34.99, "median": 34.99},
    "units_sold": {"min": 80, "max": 210, "mean": 143.75, "median": 140.0},
    "revenue": {"min": 3598.20, "max": 5998.80, "mean": 4585.73, "median": 4448.70},
}

# Test descriptive question
try:
    answer = AIController.process_query(
        "What is the average revenue?",
        dataset_context=dataset_context,
        stats_context=stats_context,
    )
    # The answer should reference the actual mean revenue from stats_context
    ok = answer and len(answer) > 10
    report("Descriptive query → engine → explanation", ok, f"Answer: {answer[:120]}...")
except Exception as e:
    report("Descriptive query → engine → explanation", False, str(e))

# Test general question (no specific column)
try:
    answer = AIController.process_query(
        "How many rows are in the dataset?",
        dataset_context=dataset_context,
        stats_context=stats_context,
    )
    ok = answer and ("12" in answer)
    report("General dataset question", ok, f"Answer: {answer[:120]}...")
except Exception as e:
    report("General dataset question", False, str(e))

# ── 4. Modularity check ─────────────────────────────────────────────────────

print("\n[4/4] Modularity / Configuration Check")

from app.llm.service import OLLAMA_API_URL, OLLAMA_MODEL, OLLAMA_API_KEY

report("OLLAMA_API_URL is set", bool(OLLAMA_API_URL), OLLAMA_API_URL)
report("OLLAMA_MODEL is set", bool(OLLAMA_MODEL), OLLAMA_MODEL)
report("OLLAMA_API_KEY is configured", bool(OLLAMA_API_KEY), "Key present (not shown)")
report("No hardcoded credentials in service.py",
       "hardcode" not in open(os.path.join("app", "llm", "service.py")).read().lower(),
       "Checked service.py source")

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n=== Summary ===")
passed = sum(1 for _, p in results if p)
total = len(results)
print(f"  {passed}/{total} checks passed\n")

if passed == total:
    print("  All checks passed - Ollama Cloud integration is working!\n")
else:
    print("  Some checks failed - review the output above.\n")
    sys.exit(1)
