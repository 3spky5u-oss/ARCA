"""
ARCA Exercise Script -- Poke the live build.

Usage: python scripts/exercise.py [--base-url http://localhost:8000]

More comprehensive than smoke_test.py. Exercises real endpoints
including admin auth, knowledge search, Qdrant, and frontend.

Exercises:
1. Health check (GET /)
2. Admin auth (POST /api/admin/login)
3. Knowledge stats (GET /api/admin/knowledge/stats)
4. Model status (GET /api/admin/models)
5. Domain info (GET /api/domain/info)
6. RAG search (POST /api/admin/knowledge/search)
7. Toggle a setting and toggle back
8. List Qdrant collections
9. Frontend health (GET http://localhost:3000)
"""

import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("[FAIL] requests library not installed. Run: pip install requests")
    sys.exit(1)


def timed_request(method, url, timeout=10, **kwargs):
    """Make a request and return (response_or_none, elapsed_ms)."""
    start = time.time()
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        elapsed = round((time.time() - start) * 1000)
        return resp, elapsed
    except requests.ConnectionError:
        elapsed = round((time.time() - start) * 1000)
        return None, elapsed
    except requests.Timeout:
        elapsed = round((time.time() - start) * 1000)
        return None, elapsed
    except Exception:
        elapsed = round((time.time() - start) * 1000)
        return None, elapsed


def report(label, passed, elapsed_ms, detail=""):
    """Print a test result line."""
    tag = "[PASS]" if passed else "[FAIL]"
    line = f"  {tag} {label} ({elapsed_ms}ms)"
    if detail:
        line += f"  {detail}"
    print(line)
    return passed


def main():
    parser = argparse.ArgumentParser(description="ARCA Exercise Script")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--frontend-url",
        default="http://localhost:3000",
        help="Frontend base URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant base URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--admin-password",
        default=None,
        help="Admin password for authenticated tests (skipped if not provided)",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    frontend = args.frontend_url.rstrip("/")
    qdrant = args.qdrant_url.rstrip("/")

    print()
    print("  ARCA Exercise Script")
    print(f"  Backend:  {base}")
    print(f"  Frontend: {frontend}")
    print(f"  Qdrant:   {qdrant}")
    print(f"  {'=' * 50}")
    print()

    results = []
    admin_token = None

    # --- 1. Health check ---
    resp, ms = timed_request("GET", f"{base}/api/health")
    if resp is None:
        ok = report("Health check (GET /api/health)", False, ms, "Connection refused")
    elif resp.status_code == 200:
        ok = report("Health check (GET /api/health)", True, ms)
    else:
        ok = report("Health check (GET /api/health)", False, ms, f"HTTP {resp.status_code}")
    results.append(ok)

    # --- 2. Admin auth ---
    if args.admin_password:
        resp, ms = timed_request(
            "POST",
            f"{base}/api/admin/login",
            json={"password": args.admin_password},
        )
        if resp is None:
            ok = report("Admin auth (POST /api/admin/login)", False, ms, "Connection refused")
        elif resp.status_code == 200:
            try:
                data = resp.json()
                admin_token = data.get("token")
                ok = report("Admin auth (POST /api/admin/login)", True, ms)
            except Exception:
                ok = report("Admin auth (POST /api/admin/login)", False, ms, "Invalid JSON response")
        else:
            ok = report("Admin auth (POST /api/admin/login)", False, ms, f"HTTP {resp.status_code}")
        results.append(ok)
    else:
        print("  [SKIP] Admin auth (POST /api/admin/login)  No --admin-password provided")

    # Helper for admin-authenticated requests
    def admin_headers():
        if admin_token:
            return {"Authorization": f"Bearer {admin_token}"}
        return {}

    # --- 3. Knowledge stats ---
    if admin_token:
        resp, ms = timed_request("GET", f"{base}/api/admin/knowledge/stats", headers=admin_headers())
        if resp is None:
            ok = report("Knowledge stats (GET /api/admin/knowledge/stats)", False, ms, "Connection refused")
        elif resp.status_code == 200:
            ok = report("Knowledge stats (GET /api/admin/knowledge/stats)", True, ms)
        else:
            ok = report("Knowledge stats (GET /api/admin/knowledge/stats)", False, ms, f"HTTP {resp.status_code}")
        results.append(ok)
    else:
        print("  [SKIP] Knowledge stats  Admin auth required")

    # --- 4. Model status ---
    resp, ms = timed_request("GET", f"{base}/api/admin/models", headers=admin_headers())
    if resp is None:
        ok = report("Model status (GET /api/admin/models)", False, ms, "Connection refused")
    elif resp.status_code == 200:
        ok = report("Model status (GET /api/admin/models)", True, ms)
    elif resp.status_code == 401:
        print("  [SKIP] Model status (GET /api/admin/models)  Admin auth required")
        ok = None
    else:
        ok = report("Model status (GET /api/admin/models)", False, ms, f"HTTP {resp.status_code}")
    if ok is not None:
        results.append(ok)

    # --- 5. Domain info ---
    resp, ms = timed_request("GET", f"{base}/api/domain/info")
    if resp is None:
        ok = report("Domain info (GET /api/domain/info)", False, ms, "Connection refused")
    elif resp.status_code == 200:
        ok = report("Domain info (GET /api/domain/info)", True, ms)
    else:
        ok = report("Domain info (GET /api/domain/info)", False, ms, f"HTTP {resp.status_code}")
    results.append(ok)

    # --- 6. RAG search ---
    if admin_token:
        resp, ms = timed_request(
            "POST",
            f"{base}/api/admin/knowledge/search",
            headers=admin_headers(),
            json={"query": "test query", "top_k": 3},
        )
        if resp is None:
            ok = report("RAG search (POST /api/admin/knowledge/search)", False, ms, "Connection refused")
        elif resp.status_code == 200:
            ok = report("RAG search (POST /api/admin/knowledge/search)", True, ms)
        else:
            ok = report("RAG search (POST /api/admin/knowledge/search)", False, ms, f"HTTP {resp.status_code}")
        results.append(ok)
    else:
        print("  [SKIP] RAG search  Admin auth required")

    # --- 7. Toggle a setting and toggle back ---
    if admin_token:
        # Read current config
        resp, ms_read = timed_request("GET", f"{base}/api/admin/config", headers=admin_headers())
        if resp is not None and resp.status_code == 200:
            try:
                config_data = resp.json()
                original_temp = config_data.get("temperature", 0.7)
                # Toggle temperature slightly
                toggled_temp = 0.71 if original_temp == 0.7 else 0.7
                resp2, ms_write = timed_request(
                    "PUT",
                    f"{base}/api/admin/config",
                    headers=admin_headers(),
                    json={"temperature": toggled_temp},
                )
                if resp2 is not None and resp2.status_code == 200:
                    # Toggle back
                    resp3, ms_restore = timed_request(
                        "PUT",
                        f"{base}/api/admin/config",
                        headers=admin_headers(),
                        json={"temperature": original_temp},
                    )
                    if resp3 is not None and resp3.status_code == 200:
                        total_ms = ms_read + ms_write + ms_restore
                        ok = report("Config toggle (GET+PUT+PUT /api/admin/config)", True, total_ms, "Toggled temperature and restored")
                    else:
                        ok = report("Config toggle (restore)", False, ms_restore, "Failed to restore original value")
                else:
                    status = resp2.status_code if resp2 else "no response"
                    ok = report("Config toggle (write)", False, ms_write, f"HTTP {status}")
            except Exception as e:
                ok = report("Config toggle", False, ms_read, str(e))
        else:
            status = resp.status_code if resp else "no response"
            ok = report("Config toggle (read)", False, ms_read, f"HTTP {status}")
        results.append(ok)
    else:
        print("  [SKIP] Config toggle  Admin auth required")

    # --- 8. List Qdrant collections ---
    resp, ms = timed_request("GET", f"{qdrant}/collections")
    if resp is None:
        ok = report("Qdrant collections (GET /collections)", False, ms, "Connection refused")
    elif resp.status_code == 200:
        try:
            data = resp.json()
            count = len(data.get("result", {}).get("collections", []))
            ok = report("Qdrant collections (GET /collections)", True, ms, f"{count} collection(s)")
        except Exception:
            ok = report("Qdrant collections (GET /collections)", True, ms)
    else:
        ok = report("Qdrant collections (GET /collections)", False, ms, f"HTTP {resp.status_code}")
    results.append(ok)

    # --- 9. Frontend health ---
    resp, ms = timed_request("GET", frontend)
    if resp is None:
        ok = report("Frontend health (GET /)", False, ms, "Connection refused")
    elif resp.status_code == 200:
        ok = report("Frontend health (GET /)", True, ms)
    else:
        ok = report("Frontend health (GET /)", False, ms, f"HTTP {resp.status_code}")
    results.append(ok)

    # --- Summary ---
    print()
    print(f"  {'=' * 50}")
    total = len(results)
    passed = sum(1 for r in results if r)
    failed = total - passed
    print(f"  {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print("  -- all clear")
    print()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
