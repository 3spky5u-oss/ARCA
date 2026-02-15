"""
ARCA Smoke Test — Boot-and-hit validation.

Hits every major endpoint and reports pass/fail.
Run after any wave, deploy, or rebuild.

Usage:
    python scripts/smoke_test.py              # Default: localhost:8000
    python scripts/smoke_test.py --host X     # Custom host
    python scripts/smoke_test.py --frontend   # Also test frontend (port 3000)
"""

import sys
import argparse
import urllib.request
import urllib.error
import time


def hit(url: str, method: str = "GET", timeout: int = 10) -> dict:
    """Hit an endpoint, return status info."""
    start = time.time()
    try:
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = round((time.time() - start) * 1000)
            return {"url": url, "status": resp.status, "ok": True, "ms": elapsed}
    except urllib.error.HTTPError as e:
        elapsed = round((time.time() - start) * 1000)
        return {"url": url, "status": e.code, "ok": False, "ms": elapsed, "error": str(e)}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"url": url, "status": 0, "ok": False, "ms": elapsed, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="ARCA Smoke Test")
    parser.add_argument("--host", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--frontend", action="store_true", help="Also test frontend")
    parser.add_argument("--frontend-host", default="http://localhost:3000", help="Frontend base URL")
    args = parser.parse_args()

    base = args.host.rstrip("/")

    # Backend endpoints to test (public, no auth required)
    endpoints = [
        ("GET", "/health"),
        ("GET", "/api/domain"),
        ("GET", "/api/instance"),
        ("GET", "/api/status"),
        ("GET", "/api/admin/auth/status"),
    ]

    # Frontend endpoints (if requested)
    frontend_endpoints = [
        ("GET", "/"),
        ("GET", "/admin"),
    ]

    print(f"\n  ARCA Smoke Test")
    print(f"  Backend: {base}")
    if args.frontend:
        print(f"  Frontend: {args.frontend_host}")
    print(f"  {'=' * 50}\n")

    results = []
    failures = 0

    # Backend tests
    for method, path in endpoints:
        result = hit(f"{base}{path}", method)
        status = "PASS" if result["ok"] else "FAIL"
        if not result["ok"]:
            failures += 1
        icon = " + " if result["ok"] else " X "
        print(f"  {icon} {status}  {method} {path}  ({result['ms']}ms) {result.get('status', '')}")
        if not result["ok"] and result.get("error"):
            print(f"         {result['error']}")
        results.append(result)

    # Frontend tests
    if args.frontend:
        print()
        fbase = args.frontend_host.rstrip("/")
        for method, path in frontend_endpoints:
            result = hit(f"{fbase}{path}", method)
            status = "PASS" if result["ok"] else "FAIL"
            if not result["ok"]:
                failures += 1
            icon = " + " if result["ok"] else " X "
            print(f"  {icon} {status}  {method} {path}  ({result['ms']}ms)")
            if not result["ok"] and result.get("error"):
                print(f"         {result['error']}")
            results.append(result)

    # Summary
    total = len(results)
    passed = total - failures
    print(f"\n  {'=' * 50}")
    print(f"  {passed}/{total} passed", end="")
    if failures:
        print(f"  ({failures} failed)")
    else:
        print(f"  — all clear")
    print()

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
