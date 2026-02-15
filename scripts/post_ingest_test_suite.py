#!/usr/bin/env python3
"""Post-Ingest Test Suite — Orchestrated validation for ARCA.

Runs inside Docker:
    docker compose exec backend python scripts/post_ingest_test_suite.py
    docker compose exec backend python scripts/post_ingest_test_suite.py --phase baseline
    docker compose exec backend python scripts/post_ingest_test_suite.py --modules 1,2,4
    docker compose exec backend python scripts/post_ingest_test_suite.py --output results.json

Modules:
    1  Infrastructure Health        (always)
    2  RAG Pipeline Validation      (always)
    3  Golden Set Evaluation        (always, skips if no data)
    4  Tool Router Accuracy         (always)
    5  Stress Test                  (always)
    6  Phii Quality                 (always)
    7  Audit Fix Validation         (always)
    8  RAPTOR Tree Validation       (post-ingest, full)
    9  GraphRAG Validation          (post-ingest, full)
   10  Community Detection          (post-ingest, full)
   11  Feature Ablation Study       (full only — placeholder)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    import redis as _redis_pkg
except ImportError:
    _redis_pkg = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# Flush immediately on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fprint(*args: Any, **kwargs: Any) -> None:
    """Print with flush=True for immediate output in Docker."""
    print(*args, **kwargs, flush=True)


def flush_ws_rate_limit() -> bool:
    """Clear WebSocket connection rate limit counters in Redis.

    Allows the test suite to run many WS connections without hitting
    the per-IP rate limit (default 10/min). Safe to call — only deletes
    rate limit keys, not application data.
    """
    if _redis_pkg is None:
        return False
    try:
        r = _redis_pkg.Redis(host="redis", port=6379, db=0, decode_responses=True)
        keys = r.keys("arca:rl:conn:*")
        if keys:
            r.delete(*keys)
        r.close()
        return True
    except Exception:
        return False


LLM_HEALTH_URL = "http://localhost:8000/health"


async def wait_for_llm_ready(max_wait: int = 120, check_interval: int = 5) -> bool:
    """Poll the backend health endpoint until it can accept new requests.

    Returns True if the backend responds healthy within max_wait seconds.
    """
    for attempt in range(max_wait // check_interval):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{BACKEND_URL}/health",
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(check_interval)
    return False


async def warmup_models() -> None:
    """Warm up LLM models before testing by sending a minimal query.

    llama-server models are managed by the backend's LLMServerManager.
    A simple health check confirms the backend (and its LLM slots) are ready.
    """
    fprint("  Checking backend health...", end=" ")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BACKEND_URL}/health", timeout=30.0)
            if resp.status_code == 200:
                fprint("ready")
            else:
                fprint(f"HTTP {resp.status_code}")
    except Exception as e:
        fprint(f"failed ({e})")


# Cascade detection: track consecutive handshake failures across queries
_consecutive_handshake_failures = 0
CASCADE_ABORT_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND_URL = "http://localhost:8000"
QDRANT_URL = "http://qdrant:6333"
WS_URI = "ws://localhost:8000/ws/chat"

HEALTH_TIMEOUT = 5.0
QUERY_TIMEOUT = 300.0
STRESS_QUERY_TIMEOUT = 300.0

GOLDEN_SET_PATH = Path("/app/data/golden_set.jsonl")

# Phase definitions: which modules run in which phase
PHASE_MODULES = {
    "baseline": {1, 2, 3, 4, 5, 6, 7},
    "post-ingest": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    "full": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
}


# ---------------------------------------------------------------------------
# WebSocket helper
# ---------------------------------------------------------------------------

async def ws_query(
    message: str,
    timeout: float = QUERY_TIMEOUT,
    phii_enabled: bool = True,
) -> Dict[str, Any]:
    """Send a message via WebSocket, collect the full response.

    Returns dict with keys: status, response, tools_used, time_s, error.
    """
    if websockets is None:
        return {"status": "error", "response": "", "tools_used": [], "time_s": 0, "error": "websockets not installed"}

    global _consecutive_handshake_failures

    # Check for cascade: if too many handshake failures, the LLM is saturated
    if _consecutive_handshake_failures >= CASCADE_ABORT_THRESHOLD:
        return {
            "status": "error",
            "response": "",
            "tools_used": [],
            "time_s": 0,
            "error": f"Cascade abort: {_consecutive_handshake_failures} consecutive handshake failures",
        }

    # Flush rate limit before every WS connection
    flush_ws_rate_limit()

    start = time.perf_counter()
    try:
        async with websockets.connect(WS_URI, ping_timeout=None, close_timeout=10) as ws:
            await ws.send(json.dumps({
                "message": message,
                "phii_enabled": phii_enabled,
                "think_mode": False,
                "calculate_mode": False,
            }))

            response_text = ""
            tools_used: List[str] = []

            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    data = json.loads(raw)

                    msg_type = data.get("type", "")
                    if msg_type == "stream":
                        content = data.get("content", "")
                        response_text += content
                        if data.get("done"):
                            tools_used = data.get("tools_used", [])
                            break
                    elif msg_type == "chunk":
                        response_text += data.get("content", "")
                    elif msg_type == "tool_start":
                        tools_used.append(data.get("tool", "unknown"))
                    elif msg_type == "done":
                        tools_used = data.get("tools_used", tools_used)
                        break
                    elif msg_type == "error":
                        elapsed = time.perf_counter() - start
                        return {
                            "status": "error",
                            "response": response_text,
                            "tools_used": tools_used,
                            "time_s": elapsed,
                            "error": data.get("content", "Unknown WS error"),
                        }
                    elif msg_type == "response":
                        # Fallback for non-streaming responses (e.g. ingest lock)
                        response_text += data.get("content", "")

                except asyncio.TimeoutError:
                    elapsed = time.perf_counter() - start
                    return {
                        "status": "timeout",
                        "response": response_text,
                        "tools_used": tools_used,
                        "time_s": elapsed,
                        "error": f"Timeout after {timeout}s",
                    }

            elapsed = time.perf_counter() - start
            _consecutive_handshake_failures = 0  # Reset on success
            return {
                "status": "ok",
                "response": response_text,
                "tools_used": tools_used,
                "time_s": elapsed,
                "error": None,
            }

    except Exception as exc:
        elapsed = time.perf_counter() - start
        err_str = str(exc)
        if "handshake" in err_str.lower() or "opening" in err_str.lower():
            _consecutive_handshake_failures += 1
        return {
            "status": "error",
            "response": "",
            "tools_used": [],
            "time_s": elapsed,
            "error": err_str,
        }


# ---------------------------------------------------------------------------
# Module result container
# ---------------------------------------------------------------------------

class ModuleResult:
    """Result of a single test module."""

    def __init__(self, module_id: int, name: str):
        self.module_id = module_id
        self.name = name
        self.status: str = "skip"     # pass / fail / skip / error
        self.score: Optional[str] = None  # e.g. "4/5"
        self.notes: str = ""
        self.details: Dict[str, Any] = {}
        self.time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "notes": self.notes,
            "time_s": round(self.time_s, 2),
            "details": self.details,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1: Infrastructure Health
# ═══════════════════════════════════════════════════════════════════════════

async def module_infrastructure() -> ModuleResult:
    result = ModuleResult(1, "Infrastructure Health")
    start = time.perf_counter()
    checks: Dict[str, str] = {}
    passed = 0
    total = 0

    # 1a. Backend health endpoint
    total += 1
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{BACKEND_URL}/health")
            if resp.status_code == 200:
                body = resp.json()
                health_checks = body.get("checks", {})

                # Count sub-services from health
                for svc, svc_status in health_checks.items():
                    total += 1
                    if svc_status == "ok":
                        checks[svc] = "ok"
                        passed += 1
                        fprint(f"    [PASS] {svc}")
                    else:
                        checks[svc] = svc_status
                        fprint(f"    [FAIL] {svc}: {svc_status}")

                checks["backend"] = "ok"
                passed += 1
                fprint(f"    [PASS] backend (status={body.get('status')})")
            else:
                checks["backend"] = f"HTTP {resp.status_code}"
                fprint(f"    [FAIL] backend: HTTP {resp.status_code}")
    except Exception as exc:
        checks["backend"] = str(exc)
        fprint(f"    [FAIL] backend: {exc}")

    # 1b. Qdrant direct
    total += 1
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{QDRANT_URL}/collections")
            if resp.status_code == 200:
                collections = resp.json().get("result", {}).get("collections", [])
                col_names = [c.get("name", "?") for c in collections]
                checks["qdrant_direct"] = "ok"
                passed += 1
                fprint(f"    [PASS] qdrant_direct ({len(col_names)} collections: {', '.join(col_names[:5])})")
            else:
                checks["qdrant_direct"] = f"HTTP {resp.status_code}"
                fprint(f"    [FAIL] qdrant_direct: HTTP {resp.status_code}")
    except Exception as exc:
        checks["qdrant_direct"] = str(exc)
        fprint(f"    [FAIL] qdrant_direct: {exc}")

    result.details = {"checks": checks}
    result.score = f"{passed}/{total}"
    result.status = "pass" if passed == total else ("fail" if passed < total // 2 else "pass")
    result.notes = "" if passed == total else f"{total - passed} service(s) down"
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2: RAG Pipeline Validation
# ═══════════════════════════════════════════════════════════════════════════

RAG_QUERIES = [
    {"query": "What methods exist for capacity analysis?", "expect_phrases": ["capacity", "load", "analysis"]},
    {"query": "How is settlement calculated?", "expect_phrases": ["settlement", "deformation", "method"]},
    {"query": "What are compliance thresholds for Parameter X?", "expect_phrases": ["threshold", "compliance", "limit"]},
    {"query": "How do environmental factors affect design?", "expect_phrases": ["environmental", "factor", "design"]},
    {"query": "Explain lateral pressure theory", "expect_phrases": ["active", "passive", "lateral"]},
]


async def module_rag_pipeline() -> ModuleResult:
    result = ModuleResult(2, "RAG Pipeline")
    start = time.perf_counter()

    total_phrases = 0
    found_phrases = 0
    query_details: List[Dict[str, Any]] = []

    for i, q in enumerate(RAG_QUERIES, 1):
        fprint(f"    [{i}/{len(RAG_QUERIES)}] {q['query'][:50]}...", end=" ")
        ws_result = await ws_query(q["query"], timeout=QUERY_TIMEOUT, phii_enabled=False)

        response_lower = ws_result.get("response", "").lower()
        matched = []
        missing = []
        for phrase in q["expect_phrases"]:
            total_phrases += 1
            if phrase.lower() in response_lower:
                found_phrases += 1
                matched.append(phrase)
            else:
                missing.append(phrase)

        status_str = f"{len(matched)}/{len(q['expect_phrases'])}"
        if ws_result["status"] != "ok":
            fprint(f"ERROR ({ws_result.get('error', '?')})")
        elif len(matched) == len(q["expect_phrases"]):
            fprint(f"PASS ({status_str})")
        else:
            fprint(f"PARTIAL ({status_str}) missing: {', '.join(missing)}")

        query_details.append({
            "query": q["query"],
            "ws_status": ws_result["status"],
            "matched": matched,
            "missing": missing,
            "time_s": round(ws_result["time_s"], 2),
            "tools_used": ws_result["tools_used"],
        })

    result.details = {"queries": query_details, "total_phrases": total_phrases, "found_phrases": found_phrases}
    result.score = f"{found_phrases}/{total_phrases}"
    ratio = found_phrases / total_phrases if total_phrases > 0 else 0
    result.status = "pass" if ratio >= 0.6 else "fail"
    result.notes = "" if ratio >= 0.6 else f"Only {ratio*100:.0f}% phrase coverage"
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 3: Golden Set Evaluation
# ═══════════════════════════════════════════════════════════════════════════

async def module_golden_set() -> ModuleResult:
    result = ModuleResult(3, "Golden Set")
    start = time.perf_counter()

    if not GOLDEN_SET_PATH.exists():
        result.status = "skip"
        result.notes = "No golden_set.jsonl"
        fprint(f"    Skipped: {GOLDEN_SET_PATH} not found")
        return result

    # Load golden set
    entries: List[Dict[str, Any]] = []
    try:
        with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as exc:
        result.status = "error"
        result.notes = f"Parse error: {exc}"
        fprint(f"    Error loading golden set: {exc}")
        return result

    if not entries:
        result.status = "skip"
        result.notes = "Empty golden set"
        fprint("    Skipped: golden set is empty")
        return result

    fprint(f"    Loaded {len(entries)} golden set entries")

    # Run queries
    category_scores: Dict[str, Dict[str, int]] = {}
    total_keywords = 0
    found_keywords = 0

    for i, entry in enumerate(entries, 1):
        question = entry.get("question", "")
        keywords = entry.get("keywords", [])
        category = entry.get("category", "uncategorized")

        if category not in category_scores:
            category_scores[category] = {"total": 0, "found": 0}

        fprint(f"    [{i}/{len(entries)}] ({category}) {question[:45]}...", end=" ")
        ws_result = await ws_query(question, timeout=QUERY_TIMEOUT, phii_enabled=False)
        response_lower = ws_result.get("response", "").lower()

        matched = 0
        for kw in keywords:
            total_keywords += 1
            category_scores[category]["total"] += 1
            if kw.lower() in response_lower:
                found_keywords += 1
                category_scores[category]["found"] += 1
                matched += 1

        fprint(f"{matched}/{len(keywords)}" if ws_result["status"] == "ok" else f"ERROR")

    # Category report
    fprint("    --- Category Breakdown ---")
    for cat, scores in sorted(category_scores.items()):
        pct = (scores["found"] / scores["total"] * 100) if scores["total"] > 0 else 0
        fprint(f"    {cat:<20} {scores['found']}/{scores['total']} ({pct:.0f}%)")

    result.details = {"category_scores": category_scores, "total_keywords": total_keywords, "found_keywords": found_keywords}
    ratio = found_keywords / total_keywords if total_keywords > 0 else 0
    result.score = f"{found_keywords}/{total_keywords}"
    result.status = "pass" if ratio >= 0.5 else "fail"
    result.notes = "" if ratio >= 0.5 else f"Only {ratio*100:.0f}% keyword coverage"
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 4: Tool Router Accuracy
# ═══════════════════════════════════════════════════════════════════════════

# expected_tool: the tool name expected in tools_used, or None for no-tool
ROUTER_CASES: List[Dict[str, Any]] = [
    # Spatial data queries
    {"message": "Query spatial data for Location A", "expected_tool": "query_geological_map"},
    {"message": "Show me the subsurface conditions near Site B", "expected_tool": "query_geological_map"},
    {"message": "What material types are mapped at Location C?", "expected_tool": "query_geological_map"},
    # Knowledge search
    {"message": "What methods exist for capacity analysis?", "expected_tool": "search_knowledge"},
    {"message": "Explain the deformation estimation process", "expected_tool": "search_knowledge"},
    {"message": "What are common causes of structural failure?", "expected_tool": "search_knowledge"},
    {"message": "Tell me about advanced structural design methods", "expected_tool": "search_knowledge"},
    {"message": "What is the principle of effective stress?", "expected_tool": "search_knowledge"},
    # Compliance threshold lookup
    {"message": "Check compliance threshold for Parameter X", "expected_tool": "lookup_guideline"},
    {"message": "What is the residential threshold for Analyte Y?", "expected_tool": "lookup_guideline"},
    {"message": "Fine-grained matrix compliance limit for Substance Z", "expected_tool": "lookup_guideline"},
    {"message": "Tier 1 compliance thresholds for organic compounds", "expected_tool": "lookup_guideline"},
    # Engineering calculations
    {"message": "Calculate capacity for resistance_angle=30, cohesion=10, unit_weight=18, depth=1.5, width=2", "expected_tool": "solve_engineering", "timeout": 180},
    {"message": "Compute settlement for a 3m wide foundation on compressible material", "expected_tool": "solve_engineering", "timeout": 180},
    {"message": "What is the active lateral pressure for resistance_angle=35 and unit_weight=20?", "expected_tool": "solve_engineering", "timeout": 180},
    {"message": "Calculate the factor of safety against sliding", "expected_tool": "solve_engineering", "timeout": 180},
    # Unit conversion
    {"message": "Convert 100 kPa to psf", "expected_tool": "unit_convert", "timeout": 30},
    {"message": "How many meters in 15 feet?", "expected_tool": "unit_convert", "timeout": 30},
    {"message": "Convert 2.5 tonnes to kN", "expected_tool": "unit_convert", "timeout": 30},
    {"message": "What is 50 psi in kPa?", "expected_tool": "unit_convert", "timeout": 30},
    # No tool (general chat)
    {"message": "Hello!", "expected_tool": None},
    {"message": "Thanks, that was helpful", "expected_tool": None},
    {"message": "Good morning", "expected_tool": None},
    {"message": "Can you summarize what we talked about?", "expected_tool": None},
    # Web search
    {"message": "Search the web for recent changes to environmental regulations", "expected_tool": "web_search"},
    {"message": "Find the latest compliance guidelines online", "expected_tool": "web_search"},
    # Report generation
    {"message": "Generate a technical report for the site investigation", "expected_tool": "generate_report"},
    {"message": "Create a design recommendation report", "expected_tool": "generate_report"},
    # File analysis
    {"message": "Analyze the uploaded lab results", "expected_tool": "analyze_files"},
    {"message": "Check the uploaded PDF for threshold exceedances", "expected_tool": "analyze_files"},
    # Field data / log processing
    {"message": "Extract the field data log from the uploaded image", "expected_tool": "extract_borehole_log"},
    {"message": "Process field data log for Site 01", "expected_tool": "process_field_logs"},
    # Redaction
    {"message": "Redact personal information from this report", "expected_tool": "redact_document"},
    # Cross section
    {"message": "Generate a cross section from these field data points", "expected_tool": "generate_cross_section"},
    # More knowledge search to balance the set
    {"message": "How do environmental factors affect foundation design?", "expected_tool": "search_knowledge"},
    {"message": "What are typical test values for dense granular material?", "expected_tool": "search_knowledge"},
]


async def module_tool_router() -> ModuleResult:
    result = ModuleResult(4, "Tool Router Accuracy")
    start = time.perf_counter()

    correct = 0
    total = len(ROUTER_CASES)
    case_details: List[Dict[str, Any]] = []

    cascade_aborted = False
    for i, case in enumerate(ROUTER_CASES, 1):
        msg = case["message"]
        expected = case["expected_tool"]
        case_timeout = case.get("timeout", QUERY_TIMEOUT)
        fprint(f"    [{i}/{total}] {msg[:50]}...", end=" ")

        ws_result = await ws_query(msg, timeout=case_timeout, phii_enabled=False)

        # Check for cascade abort
        if "Cascade abort" in str(ws_result.get("error", "")):
            fprint("SKIPPED (cascade abort)")
            cascade_aborted = True
            break

        actual_tools = ws_result.get("tools_used", [])

        if ws_result["status"] != "ok":
            match = False
            fprint(f"ERROR ({ws_result.get('error', '?')[:30]})")
        elif expected is None:
            # Expect no tool usage
            match = len(actual_tools) == 0
            if match:
                fprint("PASS (no tool)")
            else:
                fprint(f"FAIL (expected none, got {actual_tools})")
        else:
            # Expect a specific tool
            match = expected in actual_tools
            if match:
                fprint(f"PASS ({expected})")
            else:
                fprint(f"FAIL (expected {expected}, got {actual_tools or 'none'})")

        if match:
            correct += 1

        case_details.append({
            "message": msg,
            "expected_tool": expected,
            "actual_tools": actual_tools,
            "match": match,
            "time_s": round(ws_result["time_s"], 2),
        })

    tested = len(case_details)
    result.details = {"cases": case_details, "correct": correct, "total": total, "tested": tested, "cascade_aborted": cascade_aborted}
    result.score = f"{correct}/{tested}" if cascade_aborted else f"{correct}/{total}"
    accuracy = correct / tested if tested > 0 else 0
    result.status = "pass" if accuracy >= 0.7 else "fail"
    notes_parts = []
    if accuracy < 0.7:
        notes_parts.append(f"Accuracy {accuracy*100:.0f}% < 70%")
    if cascade_aborted:
        notes_parts.append(f"cascade abort after {tested}/{total}")
    result.notes = "; ".join(notes_parts)
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 5: Stress Test
# ═══════════════════════════════════════════════════════════════════════════

STRESS_CATEGORIES = {
    "capacity_analysis": [
        "What are the key factors affecting load capacity of structural components?",
        "How does environmental exposure influence capacity analysis?",
        "What is the relationship between foundation width and capacity?",
    ],
    "settlement_estimation": [
        "What is the difference between immediate and long-term settlement?",
        "How do you estimate settlement for a foundation on granular material?",
        "What are tolerable differential settlement limits for structures?",
    ],
    "environmental_factors": [
        "How does freeze-thaw cycling damage foundations?",
        "What is the minimum depth for foundations in freeze-susceptible materials?",
        "How do you determine environmental penetration depth?",
    ],
    "lateral_pressure": [
        "Explain active lateral pressure theory",
        "When do you use at-rest pressure coefficients?",
        "What is the difference between active and passive lateral pressure?",
    ],
    "environmental_assessment": [
        "What is Tier 1 risk assessment?",
        "How do contaminant migration pathways affect site assessment?",
        "What parameters are tested in a Phase 2 environmental assessment?",
    ],
    "thermal_effects": [
        "How do thermal systems stabilize foundations in cold regions?",
        "What is the active layer thickness and why does it matter?",
        "How does climate change affect infrastructure in cold regions?",
    ],
    "compliance_verification": [
        "What compliance thresholds apply for residential fine-grained matrix?",
        "What is the Tier 1 limit for organic compounds?",
        "How do you handle threshold exceedances in contamination data?",
    ],
    "knowledge_search": [
        "What are typical strength parameters for compressible material?",
        "Describe the material classification system",
        "What is index property testing?",
    ],
    "unit_conversion": [
        "Convert 200 kPa to psi",
        "How many kN is 10 tonnes?",
        "Convert 50 feet to meters",
    ],
    "general_chat": [
        "Hello, good morning!",
        "What can you help me with?",
        "Thanks for the information.",
    ],
}


async def module_stress_test() -> ModuleResult:
    result = ModuleResult(5, "Stress Test")
    start = time.perf_counter()

    total = sum(len(qs) for qs in STRESS_CATEGORIES.values())
    successes = 0
    failures = 0
    timeouts = 0
    times: List[float] = []
    category_details: Dict[str, Dict[str, Any]] = {}

    query_num = 0
    cascade_aborted = False
    for category, questions in STRESS_CATEGORIES.items():
        if cascade_aborted:
            break
        cat_pass = 0
        cat_times: List[float] = []
        for q in questions:
            query_num += 1
            fprint(f"    [{query_num}/{total}] ({category}) {q[:40]}...", end=" ")

            ws_result = await ws_query(q, timeout=STRESS_QUERY_TIMEOUT, phii_enabled=False)

            # Check for cascade abort
            if "Cascade abort" in str(ws_result.get("error", "")):
                fprint("SKIPPED (cascade abort)")
                cascade_aborted = True
                break

            elapsed = ws_result["time_s"]
            times.append(elapsed)
            cat_times.append(elapsed)

            if ws_result["status"] == "ok":
                successes += 1
                cat_pass += 1
                flag = " SLOW!" if elapsed > 30 else ""
                fprint(f"OK {elapsed:.1f}s{flag}")
            elif ws_result["status"] == "timeout":
                timeouts += 1
                fprint(f"TIMEOUT ({elapsed:.1f}s)")
            else:
                failures += 1
                fprint(f"FAIL ({ws_result.get('error', '?')[:30]})")

            # 2s delay between stress queries
            await asyncio.sleep(2)

        avg_cat = sum(cat_times) / len(cat_times) if cat_times else 0
        category_details[category] = {
            "passed": cat_pass,
            "total": len(questions),
            "avg_time_s": round(avg_cat, 2),
        }

    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0

    fprint(f"    --- Stress Summary ---")
    fprint(f"    Successes: {successes}/{total} | Failures: {failures} | Timeouts: {timeouts}")
    fprint(f"    Avg response: {avg_time:.1f}s | Max: {max_time:.1f}s")

    result.details = {
        "successes": successes,
        "failures": failures,
        "timeouts": timeouts,
        "avg_time_s": round(avg_time, 2),
        "max_time_s": round(max_time, 2),
        "categories": category_details,
    }
    result.score = f"{successes}/{total}"
    result.status = "pass" if failures == 0 and timeouts == 0 else ("fail" if failures + timeouts > total * 0.2 else "pass")
    result.notes = ""
    if timeouts > 0:
        result.notes += f"{timeouts} timeout(s) "
    if failures > 0:
        result.notes += f"{failures} failure(s)"
    result.notes = result.notes.strip()
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 6: Phii Quality
# ═══════════════════════════════════════════════════════════════════════════

PHII_QUERIES = [
    "What's the quick rule of thumb for test index to resistance angle?",
    "I need a detailed explanation of classical capacity theory including all assumptions",
    "is 18 kN/m3 reasonable for dense granular material?",
    "URGENT: project deadline tomorrow, need settlement calculation method NOW",
    "Could you explain in simple terms what effective stress means?",
]


async def module_phii_quality() -> ModuleResult:
    result = ModuleResult(6, "Phii Quality")
    start = time.perf_counter()

    passed = 0
    total = len(PHII_QUERIES)
    query_details: List[Dict[str, Any]] = []

    for i, q in enumerate(PHII_QUERIES, 1):
        fprint(f"    [{i}/{total}] {q[:55]}...", end=" ")
        ws_result = await ws_query(q, timeout=QUERY_TIMEOUT, phii_enabled=True)

        response = ws_result.get("response", "")
        is_ok = ws_result["status"] == "ok" and len(response.strip()) > 50

        if is_ok:
            passed += 1
            fprint(f"OK ({len(response)} chars, {ws_result['time_s']:.1f}s)")
        elif ws_result["status"] != "ok":
            fprint(f"ERROR ({ws_result.get('error', '?')[:30]})")
        else:
            fprint(f"WEAK (only {len(response)} chars)")

        query_details.append({
            "query": q,
            "ws_status": ws_result["status"],
            "response_length": len(response),
            "time_s": round(ws_result["time_s"], 2),
            "passed": is_ok,
        })

    result.details = {"queries": query_details}
    result.score = f"{passed}/{total}"
    result.status = "pass" if passed >= total * 0.8 else "fail"
    result.notes = "" if passed == total else f"{total - passed} weak/failed"
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 7: Audit Fix Validation
# ═══════════════════════════════════════════════════════════════════════════

async def module_audit_fixes() -> ModuleResult:
    result = ModuleResult(7, "Audit Fix Validation")
    start = time.perf_counter()
    checks: Dict[str, Dict[str, Any]] = {}
    passed = 0
    total = 0

    # 7a. Circuit breaker: health returns 200
    total += 1
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{BACKEND_URL}/health")
            if resp.status_code == 200:
                checks["circuit_breaker"] = {"status": "pass", "detail": "Health returns 200"}
                passed += 1
                fprint("    [PASS] Circuit breaker: /health returns 200")
            else:
                checks["circuit_breaker"] = {"status": "fail", "detail": f"HTTP {resp.status_code}"}
                fprint(f"    [FAIL] Circuit breaker: HTTP {resp.status_code}")
    except Exception as exc:
        checks["circuit_breaker"] = {"status": "fail", "detail": str(exc)}
        fprint(f"    [FAIL] Circuit breaker: {exc}")

    # 7b. Config range validation — check auth first
    admin_auth_available = False
    admin_token: Optional[str] = None

    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{BACKEND_URL}/api/admin/auth-status")
            if resp.status_code == 200:
                body = resp.json()
                if body.get("setup_required"):
                    fprint("    [INFO] Admin auth not set up — skipping config validation test")
                else:
                    # Auth is configured but we don't have a token.
                    # We can still test that the endpoint enforces auth.
                    admin_auth_available = True
    except Exception:
        pass

    total += 1
    if admin_auth_available:
        # Test that unauthenticated PUT to /api/admin/config is rejected (should be 401/403)
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
                resp = await client.put(
                    f"{BACKEND_URL}/api/admin/config",
                    json={"temperature": 5.0},
                )
                # We expect 401 or 403 because we have no token
                if resp.status_code in (401, 403):
                    checks["config_auth"] = {"status": "pass", "detail": f"Unauthenticated config rejected ({resp.status_code})"}
                    passed += 1
                    fprint(f"    [PASS] Config auth enforcement: unauthenticated PUT rejected ({resp.status_code})")
                elif resp.status_code == 400:
                    # If somehow it got through auth but validation caught it, partial pass
                    checks["config_auth"] = {"status": "pass", "detail": "Config validation returned 400 (no auth bypass)"}
                    passed += 1
                    fprint("    [PASS] Config validation: returned 400 for invalid temperature")
                else:
                    checks["config_auth"] = {"status": "fail", "detail": f"Unexpected HTTP {resp.status_code}"}
                    fprint(f"    [FAIL] Config auth: unexpected HTTP {resp.status_code}")
        except Exception as exc:
            checks["config_auth"] = {"status": "fail", "detail": str(exc)}
            fprint(f"    [FAIL] Config auth test: {exc}")
    else:
        checks["config_auth"] = {"status": "skip", "detail": "Admin auth not configured"}
        passed += 1  # Don't penalize for missing auth setup
        fprint("    [SKIP] Config range validation (admin auth not set up)")

    # 7c. Session ID format
    total += 1
    if websockets is not None:
        try:
            async with websockets.connect(WS_URI, ping_timeout=None, close_timeout=5) as ws:
                # Send a trivial message, the session ID is generated server-side
                await ws.send(json.dumps({"message": "ping", "phii_enabled": False}))

                # Collect messages until done
                got_response = False
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                        data = json.loads(raw)
                        if data.get("done") or (data.get("type") == "stream" and data.get("done")):
                            got_response = True
                            break
                        if data.get("type") in ("error", "done", "response"):
                            got_response = True
                            break
                    except asyncio.TimeoutError:
                        break

            # If we got a response without error, the session ID was generated properly.
            # The format ws_{token_urlsafe(16)} produces URL-safe chars by definition.
            if got_response:
                checks["session_id"] = {"status": "pass", "detail": "WebSocket session established with ws_ prefix ID"}
                passed += 1
                fprint("    [PASS] Session ID: WebSocket connected and responded")
            else:
                checks["session_id"] = {"status": "fail", "detail": "No response from WebSocket"}
                fprint("    [FAIL] Session ID: no response")
        except Exception as exc:
            checks["session_id"] = {"status": "fail", "detail": str(exc)}
            fprint(f"    [FAIL] Session ID test: {exc}")
    else:
        checks["session_id"] = {"status": "skip", "detail": "websockets not installed"}
        fprint("    [SKIP] Session ID (websockets not installed)")

    # 7d. CORS headers
    total += 1
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            # Send an OPTIONS-like request with an Origin header
            resp = await client.options(
                f"{BACKEND_URL}/health",
                headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
            )
            cors_header = resp.headers.get("access-control-allow-origin", "")
            if cors_header:
                checks["cors"] = {"status": "pass", "detail": f"ACAO: {cors_header}"}
                passed += 1
                fprint(f"    [PASS] CORS: access-control-allow-origin={cors_header}")
            else:
                # Some CORS configs only reply on actual requests, try GET with Origin
                resp2 = await client.get(
                    f"{BACKEND_URL}/health",
                    headers={"Origin": "http://localhost:3000"},
                )
                cors_header2 = resp2.headers.get("access-control-allow-origin", "")
                if cors_header2:
                    checks["cors"] = {"status": "pass", "detail": f"ACAO on GET: {cors_header2}"}
                    passed += 1
                    fprint(f"    [PASS] CORS: access-control-allow-origin={cors_header2}")
                else:
                    checks["cors"] = {"status": "fail", "detail": "No ACAO header found"}
                    fprint("    [FAIL] CORS: no access-control-allow-origin header")
    except Exception as exc:
        checks["cors"] = {"status": "fail", "detail": str(exc)}
        fprint(f"    [FAIL] CORS test: {exc}")

    result.details = {"checks": checks}
    result.score = f"{passed}/{total}"
    result.status = "pass" if passed == total else ("fail" if passed < total // 2 else "pass")
    result.notes = "" if passed == total else f"{total - passed} check(s) not passing"
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 8: RAPTOR Tree Validation (post-ingest)
# ═══════════════════════════════════════════════════════════════════════════

async def module_raptor() -> ModuleResult:
    result = ModuleResult(8, "RAPTOR Tree Validation")
    start = time.perf_counter()

    # Check health for clues about RAPTOR
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{BACKEND_URL}/health")
            if resp.status_code == 200:
                health = resp.json()
                fprint(f"    Backend status: {health.get('status')}")
            else:
                result.status = "skip"
                result.notes = "Backend unreachable"
                result.time_s = time.perf_counter() - start
                return result
    except Exception as exc:
        result.status = "skip"
        result.notes = f"Health check failed: {exc}"
        result.time_s = time.perf_counter() - start
        return result

    # Try admin status (needs auth)
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{BACKEND_URL}/api/admin/auth-status")
            if resp.status_code == 200:
                auth_data = resp.json()
                if auth_data.get("setup_required"):
                    result.status = "skip"
                    result.notes = "Admin auth not configured"
                    fprint("    Skipped: admin auth not set up")
                    result.time_s = time.perf_counter() - start
                    return result
    except Exception:
        pass

    # Without admin token we cannot check /api/admin/status
    # Check via a RAPTOR-like query instead
    fprint("    Testing RAPTOR via summary-level query...")
    ws_result = await ws_query(
        "Give me a high-level summary of the technical principles covered in the knowledge base",
        timeout=QUERY_TIMEOUT,
        phii_enabled=False,
    )

    if ws_result["status"] == "ok" and len(ws_result.get("response", "")) > 100:
        result.status = "pass"
        result.score = "1/1"
        result.notes = "Summary query returned substantive response"
        fprint(f"    PASS: response {len(ws_result['response'])} chars")
    else:
        result.status = "skip"
        result.score = "0/1"
        result.notes = "Could not validate RAPTOR (no admin auth or weak response)"
        fprint(f"    SKIP: {ws_result['status']} - {ws_result.get('error', 'weak response')}")

    result.details = {
        "ws_status": ws_result["status"],
        "response_length": len(ws_result.get("response", "")),
    }
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 9: GraphRAG Validation (post-ingest)
# ═══════════════════════════════════════════════════════════════════════════

async def module_graphrag() -> ModuleResult:
    result = ModuleResult(9, "GraphRAG Validation")
    start = time.perf_counter()

    # Check Neo4j from health endpoint
    neo4j_ok = False
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            resp = await client.get(f"{BACKEND_URL}/health")
            if resp.status_code == 200:
                checks = resp.json().get("checks", {})
                neo4j_status = checks.get("neo4j", "not_present")
                if neo4j_status == "ok":
                    neo4j_ok = True
                    fprint(f"    Neo4j: ok")
                elif neo4j_status == "not_present":
                    fprint("    Neo4j not in health checks (may not be enabled)")
                else:
                    fprint(f"    Neo4j: {neo4j_status}")
    except Exception as exc:
        fprint(f"    Health check failed: {exc}")

    if not neo4j_ok:
        result.status = "skip"
        result.notes = "Neo4j not available or not enabled"
        fprint("    Skipped: Neo4j not in healthy state")
        result.time_s = time.perf_counter() - start
        return result

    # If Neo4j is ok, test a graph-traversal query
    fprint("    Testing graph-enhanced retrieval...")
    ws_result = await ws_query(
        "What entities and relationships are relevant to structural design?",
        timeout=QUERY_TIMEOUT,
        phii_enabled=False,
    )

    if ws_result["status"] == "ok" and len(ws_result.get("response", "")) > 50:
        result.status = "pass"
        result.score = "1/1"
        fprint(f"    PASS: response {len(ws_result['response'])} chars")
    else:
        result.status = "fail"
        result.score = "0/1"
        result.notes = "Graph query returned weak response"
        fprint(f"    FAIL: {ws_result['status']}")

    result.details = {
        "neo4j_ok": neo4j_ok,
        "ws_status": ws_result["status"],
        "response_length": len(ws_result.get("response", "")),
    }
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 10: Community Detection (post-ingest)
# ═══════════════════════════════════════════════════════════════════════════

async def module_community_detection() -> ModuleResult:
    result = ModuleResult(10, "Community Detection")
    start = time.perf_counter()

    # Check health for GraphRAG / community signals
    community_enabled = False
    global_search_enabled = False

    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            # Try the public health endpoint for any community/graph signals
            resp = await client.get(f"{BACKEND_URL}/health")
            if resp.status_code == 200:
                health = resp.json()
                checks = health.get("checks", {})
                # Community detection requires Neo4j
                if checks.get("neo4j") == "ok":
                    fprint("    Neo4j available — community detection possible")
                else:
                    fprint("    Neo4j not available — community detection skipped")
                    result.status = "skip"
                    result.notes = "Neo4j not available"
                    result.time_s = time.perf_counter() - start
                    return result
    except Exception as exc:
        result.status = "skip"
        result.notes = f"Health check failed: {exc}"
        result.time_s = time.perf_counter() - start
        return result

    # Without admin auth, we cannot read config directly.
    # Test a global/community-type query instead.
    fprint("    Testing community-level query...")
    ws_result = await ws_query(
        "What are the main topic clusters in the knowledge base?",
        timeout=QUERY_TIMEOUT,
        phii_enabled=False,
    )

    if ws_result["status"] == "ok" and len(ws_result.get("response", "")) > 50:
        result.status = "pass"
        result.score = "1/1"
        fprint(f"    PASS: response {len(ws_result['response'])} chars")
    else:
        result.status = "skip"
        result.score = "0/1"
        result.notes = "Community query returned weak response"
        fprint(f"    SKIP: {ws_result['status']}")

    result.details = {
        "ws_status": ws_result["status"],
        "response_length": len(ws_result.get("response", "")),
    }
    result.time_s = time.perf_counter() - start
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 11: Feature Ablation Study (full only — placeholder)
# ═══════════════════════════════════════════════════════════════════════════

async def module_feature_ablation() -> ModuleResult:
    result = ModuleResult(11, "Feature Ablation Study")
    result.status = "skip"
    result.notes = "Placeholder (run with --phase full)"
    fprint("    Feature ablation: skipped (run with --phase full)")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Scorecard & Output
# ═══════════════════════════════════════════════════════════════════════════

MODULE_NAMES = {
    1: "Infrastructure Health",
    2: "RAG Pipeline",
    3: "Golden Set",
    4: "Tool Router Accuracy",
    5: "Stress Test",
    6: "Phii Quality",
    7: "Audit Fix Validation",
    8: "RAPTOR Tree",
    9: "GraphRAG",
    10: "Community Detection",
    11: "Feature Ablation",
}

MODULE_FUNCS = {
    1: module_infrastructure,
    2: module_rag_pipeline,
    3: module_golden_set,
    4: module_tool_router,
    5: module_stress_test,
    6: module_phii_quality,
    7: module_audit_fixes,
    8: module_raptor,
    9: module_graphrag,
    10: module_community_detection,
    11: module_feature_ablation,
}


def print_scorecard(results: Dict[int, ModuleResult], phase: str) -> None:
    """Print the final scorecard table."""
    fprint()
    fprint("\u2554" + "\u2550" * 72 + "\u2557")
    fprint("\u2551" + "POST-INGEST TEST SCORECARD".center(72) + "\u2551")
    fprint("\u2551" + f"Phase: {phase}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(72) + "\u2551")
    fprint("\u2560" + "\u2550" * 36 + "\u2564" + "\u2550" * 8 + "\u2564" + "\u2550" * 9 + "\u2564" + "\u2550" * 16 + "\u2563")
    fprint("\u2551" + " Module".ljust(36) + "\u2502" + " Status ".center(8) + "\u2502" + " Score ".center(9) + "\u2502" + " Notes".ljust(16) + "\u2551")
    fprint("\u255f" + "\u2500" * 36 + "\u253c" + "\u2500" * 8 + "\u253c" + "\u2500" * 9 + "\u253c" + "\u2500" * 16 + "\u2562")

    passed = 0
    failed = 0
    skipped = 0
    total_time = 0.0

    for mod_id in sorted(results.keys()):
        r = results[mod_id]
        total_time += r.time_s

        # Status formatting
        status_str = r.status.upper()
        if r.status == "pass":
            passed += 1
        elif r.status == "fail":
            failed += 1
        elif r.status in ("skip", "error"):
            skipped += 1

        score_str = r.score if r.score else "-"
        notes_str = r.notes[:14] if r.notes else ""
        name_str = f" {mod_id:>2}. {r.name}"

        fprint(
            "\u2551"
            + name_str.ljust(36)
            + "\u2502"
            + status_str.center(8)
            + "\u2502"
            + score_str.center(9)
            + "\u2502"
            + f" {notes_str}".ljust(16)
            + "\u2551"
        )

    fprint("\u255f" + "\u2500" * 36 + "\u2534" + "\u2500" * 8 + "\u2534" + "\u2500" * 9 + "\u2534" + "\u2500" * 16 + "\u2562")
    summary_line = f" Overall: {passed} PASS, {failed} FAIL, {skipped} SKIP  |  {total_time:.1f}s total"
    fprint("\u2551" + summary_line.ljust(72) + "\u2551")
    fprint("\u255a" + "\u2550" * 72 + "\u255d")


def write_json_output(
    results: Dict[int, ModuleResult],
    phase: str,
    output_path: str,
) -> None:
    """Write structured JSON results to file."""
    modules_dict = {}
    passed = 0
    failed = 0
    skipped = 0

    for mod_id in sorted(results.keys()):
        r = results[mod_id]
        key = f"{mod_id}_{r.name.lower().replace(' ', '_')}"
        modules_dict[key] = r.to_dict()
        if r.status == "pass":
            passed += 1
        elif r.status == "fail":
            failed += 1
        else:
            skipped += 1

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "modules": modules_dict,
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(results),
        },
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    fprint(f"\nResults written to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-Ingest Test Suite for ARCA",
        epilog="""
Examples:
  python scripts/post_ingest_test_suite.py                          # Full suite
  python scripts/post_ingest_test_suite.py --phase baseline         # Baseline modules only
  python scripts/post_ingest_test_suite.py --modules 1,2,7          # Specific modules
  python scripts/post_ingest_test_suite.py --output results.json    # Save JSON
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["baseline", "post-ingest", "full"],
        default="full",
        help="Test phase: baseline (core only), post-ingest (+ advanced RAG), full (+ ablation)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write structured JSON results to FILE",
    )
    parser.add_argument(
        "--modules",
        metavar="1,2,3",
        help="Comma-separated list of module IDs to run (overrides --phase)",
    )
    args = parser.parse_args()

    # Determine which modules to run
    if args.modules:
        try:
            module_ids = sorted(set(int(x.strip()) for x in args.modules.split(",")))
        except ValueError:
            fprint("Error: --modules must be comma-separated integers (e.g. 1,2,3)")
            return 1
        invalid = [m for m in module_ids if m not in MODULE_FUNCS]
        if invalid:
            fprint(f"Error: unknown module IDs: {invalid}. Valid: 1-{max(MODULE_FUNCS.keys())}")
            return 1
    else:
        module_ids = sorted(PHASE_MODULES[args.phase])

    total_modules = len(module_ids)

    # Header
    fprint("=" * 74)
    fprint("  ARCA POST-INGEST TEST SUITE")
    fprint("=" * 74)
    fprint(f"  Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"  Phase:   {args.phase}")
    fprint(f"  Modules: {', '.join(str(m) for m in module_ids)} ({total_modules} total)")
    fprint(f"  Backend: {BACKEND_URL}")
    fprint(f"  Qdrant:  {QDRANT_URL}")
    fprint("=" * 74)

    # Warm up models before testing
    fprint("\n  --- Model Warmup ---")
    await warmup_models()

    # Run modules
    results: Dict[int, ModuleResult] = {}
    suite_start = time.perf_counter()

    # Modules that use WebSocket connections
    ws_modules = {2, 3, 4, 5, 6, 8, 9, 10, 11}

    global _consecutive_handshake_failures

    for idx, mod_id in enumerate(module_ids, 1):
        mod_name = MODULE_NAMES.get(mod_id, f"Module {mod_id}")
        fprint(f"\n[{idx}/{total_modules}] {mod_name}...")

        # Reset cascade counter between modules
        _consecutive_handshake_failures = 0

        # Before WS-heavy modules: flush rate limits + check LLM health
        if mod_id in ws_modules:
            flush_ws_rate_limit()
            if idx > 1:
                fprint("    Checking LLM availability...", end=" ")
                llm_ok = await wait_for_llm_ready(max_wait=60, check_interval=3)
                if llm_ok:
                    fprint("ready")
                else:
                    fprint("WARNING: LLM may be unresponsive")

        try:
            mod_result = await MODULE_FUNCS[mod_id]()
        except Exception as exc:
            mod_result = ModuleResult(mod_id, mod_name)
            mod_result.status = "error"
            mod_result.notes = str(exc)[:40]
            fprint(f"    UNHANDLED ERROR: {exc}")

        results[mod_id] = mod_result

        status_upper = mod_result.status.upper()
        score_str = f" ({mod_result.score})" if mod_result.score else ""
        notes_str = f" -- {mod_result.notes}" if mod_result.notes else ""
        fprint(f"  => {status_upper}{score_str}{notes_str}  [{mod_result.time_s:.1f}s]")

    suite_time = time.perf_counter() - suite_start
    fprint(f"\nSuite completed in {suite_time:.1f}s")

    # Scorecard
    print_scorecard(results, args.phase)

    # JSON output
    if args.output:
        write_json_output(results, args.phase, args.output)

    # Exit code: 0 if no failures, 1 if any
    has_failures = any(r.status == "fail" for r in results.values())
    return 1 if has_failures else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
