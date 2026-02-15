"""
ARCA Hardware Simulation Runner — cycles through hardware profiles and reports
what works, what breaks, and what degrades at each VRAM tier.

Produces a detailed report that can be fed to an agent for analysis.

Usage:
    python scripts/test_hardware_sim.py                    # All profiles, terminal output
    python scripts/test_hardware_sim.py --report sim.md    # Save markdown report
    python scripts/test_hardware_sim.py small medium       # Test specific profiles
    python scripts/test_hardware_sim.py --live             # Also hit running backend APIs
    python scripts/test_hardware_sim.py --list             # Show profile specs

Full-stack simulation (add to .env, rebuild with docker compose up -d --build):
    ARCA_SIMULATE_VRAM=8g        # 8 GB VRAM -> small profile
    ARCA_SIMULATE_VRAM=0         # CPU-only
    ARCA_SIMULATE_GPU=GTX 1070   # Optional: fake GPU name
    ARCA_SIMULATE_RAM=8          # Optional: fake RAM in GB
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error

# Backend imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# ---------------------------------------------------------------------------
# Profile scenarios — realistic hardware configs to simulate
# ---------------------------------------------------------------------------

SCENARIOS = {
    "cpu": {
        "label": "CPU-Only (Integrated / No NVIDIA)",
        "vram_mb": 0,
        "gpu_name": "None (CPU-only)",
        "ram_gb": 16,
        "cpu": "Intel i5-12400",
        "cores": 12,
        "real_world": "Office laptop, cloud VM without GPU, older desktop",
    },
    "small": {
        "label": "Entry GPU (GTX 1660 / RTX 3060 6GB)",
        "vram_mb": 6144,
        "gpu_name": "NVIDIA GeForce GTX 1660 Super",
        "ram_gb": 16,
        "cpu": "AMD Ryzen 5 5600X",
        "cores": 12,
        "real_world": "Budget gaming PC, entry workstation",
    },
    "medium": {
        "label": "Mid-Range GPU (RTX 3080 10GB / RTX 4070 12GB)",
        "vram_mb": 12288,
        "gpu_name": "NVIDIA GeForce RTX 4070",
        "ram_gb": 32,
        "cpu": "AMD Ryzen 7 7700X",
        "cores": 16,
        "real_world": "Gaming PC, mid-tier workstation",
    },
    "large": {
        "label": "High-End GPU (RTX 4090 24GB / RTX 3090 24GB)",
        "vram_mb": 24576,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "ram_gb": 64,
        "cpu": "AMD Ryzen 9 9950X3D",
        "cores": 32,
        "real_world": "High-end workstation, ML development machine",
    },
    "large_32": {
        "label": "Prosumer GPU (A5000 / Your Setup ~32GB)",
        "vram_mb": 32000,
        "gpu_name": "NVIDIA RTX A5000",
        "ram_gb": 64,
        "cpu": "AMD Ryzen 9 9950X3D",
        "cores": 32,
        "real_world": "Professional workstation, your current hardware",
    },
}

# Known model components and their VRAM cost
COMPONENT_STACK = [
    ("Embedder (Qwen3-0.6B)", 800),
    ("Reranker (BGE-v2-M3)", 600),
    ("Chat LLM 7B Q4", 5000),
    ("Chat LLM 7B Q8", 8000),
    ("Chat LLM 14B Q4", 9000),
    ("Chat LLM 14B Q8", 16000),
    ("Expert LLM 32B Q4", 20000),
    ("Vision LLM 8B Q4", 5500),
    ("Vision LLM 8B Q8", 8000),
]

# Minimum viable stack: embedder + reranker + smallest chat LLM
MIN_STACK = [
    ("Embedder", 800),
    ("Reranker", 600),
    ("Chat 7B Q4", 5000),
]
MIN_STACK_TOTAL = sum(v for _, v in MIN_STACK)

# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

def _reset():
    """Reset hardware singleton for next simulation."""
    import services.hardware as hw
    hw._hardware_info = None


def simulate_profile(name: str, scenario: dict) -> dict:
    """Run full simulation for one hardware profile. Returns structured results."""
    import services.hardware as hw

    # Set simulation env
    os.environ["ARCA_SIMULATE_VRAM"] = str(scenario["vram_mb"])
    os.environ["ARCA_SIMULATE_GPU"] = scenario["gpu_name"]
    os.environ["ARCA_SIMULATE_RAM"] = str(scenario["ram_gb"])
    os.environ["ARCA_SIMULATE_CPU"] = scenario.get("cpu", "Simulated CPU")
    os.environ["ARCA_SIMULATE_CORES"] = str(scenario.get("cores", 4))

    _reset()
    info = hw.get_hardware_info()
    profile_defaults = hw.PROFILES.get(info.profile, {})
    recs = hw.MODEL_RECOMMENDATIONS.get(info.profile, {})

    result = {
        "name": name,
        "label": scenario["label"],
        "real_world": scenario["real_world"],
        "profile": info.profile,
        "vram_total_mb": info.gpu_vram_total_mb,
        "vram_total_gb": round(info.gpu_vram_total_mb / 1024, 1) if info.gpu_vram_total_mb else 0,
        "gpu_name": info.gpu_name,
        "ram_gb": info.ram_total_gb,
        "ctx_size": profile_defaults.get("ctx_size", 0),
        "gpu_layers": profile_defaults.get("gpu_layers", 0),
        "batch_size": profile_defaults.get("batch_size", 0),
        "recommendations": recs,
        "components": [],
        "stack_analysis": {},
        "red_flags": [],
        "green_flags": [],
    }

    # Test each component against VRAM budget
    available = hw.get_vram_available_mb()
    for comp_name, comp_vram in COMPONENT_STACK:
        can_load = hw.check_vram_budget(comp_name, comp_vram)
        result["components"].append({
            "name": comp_name,
            "vram_mb": comp_vram,
            "can_load": can_load,
            "pct_of_total": round(comp_vram / scenario["vram_mb"] * 100, 1) if scenario["vram_mb"] > 0 else 0,
        })

    # Stack analysis: can we run the minimum stack?
    if scenario["vram_mb"] == 0:
        # CPU mode — everything runs on CPU, no VRAM budget
        result["stack_analysis"] = {
            "min_stack_fits": True,
            "min_stack_cost_mb": MIN_STACK_TOTAL,
            "headroom_after_min_mb": 0,
            "mode": "cpu",
            "note": "All models run on CPU (slow but functional). No VRAM budget applies.",
        }
        result["green_flags"].append("CPU-only mode is functional — ARCA will work, just slower")
        result["red_flags"].append("No GPU acceleration — expect 10-50x slower inference")
        result["red_flags"].append("Vision and expert modes likely too slow to be usable")
    else:
        headroom = available - MIN_STACK_TOTAL
        can_min = headroom >= 0
        result["stack_analysis"] = {
            "min_stack_fits": can_min,
            "min_stack_cost_mb": MIN_STACK_TOTAL,
            "headroom_after_min_mb": max(headroom, 0),
            "headroom_after_min_gb": round(max(headroom, 0) / 1024, 1),
            "mode": "gpu",
        }

        if not can_min:
            deficit = -headroom
            result["red_flags"].append(
                f"MINIMUM STACK DOES NOT FIT: need {MIN_STACK_TOTAL} MB, have {available} MB "
                f"(short {deficit} MB). System will fail to load base models."
            )
        else:
            result["green_flags"].append(
                f"Base stack fits: {MIN_STACK_TOTAL} MB used, {headroom} MB headroom"
            )

        # Can we add expert mode (swap chat for 32B)?
        expert_cost = 20000  # 32B Q4
        expert_headroom = available - 800 - 600 - expert_cost  # embed + rerank + expert
        if expert_headroom >= 0:
            result["green_flags"].append(f"Expert mode (32B Q4) fits alongside base stack")
        else:
            if info.profile in ("medium", "large"):
                result["red_flags"].append(
                    f"Expert mode requires model swap — 32B Q4 needs {expert_cost} MB, "
                    f"must unload chat model first"
                )

        # Can we add vision (on-demand)?
        vision_cost = 5500  # 8B Q4
        vision_needs_swap = (available - 800 - 600 - vision_cost) < 5000  # can't run alongside chat
        if vision_needs_swap:
            result["red_flags"].append("Vision requires swapping out chat LLM (on-demand loading)")

        # Context window impact
        if result["ctx_size"] <= 4096:
            result["red_flags"].append(
                f"Context window limited to {result['ctx_size']} tokens — "
                f"long documents may be truncated, complex queries may lose context"
            )

        # Batch size impact
        if result["batch_size"] <= 8:
            result["red_flags"].append(
                f"Batch size {result['batch_size']} — bulk operations (ingest, benchmark) will be slow"
            )

    return result


def format_result_terminal(r: dict) -> str:
    """Format a single profile result for terminal output."""
    lines = []
    vram = f"{r['vram_total_gb']} GB" if r['vram_total_gb'] else "None"
    lines.append(f"\n  {'=' * 65}")
    lines.append(f"  [{r['name'].upper()}] {r['label']}")
    lines.append(f"  {r['real_world']}")
    lines.append(f"  {'=' * 65}")
    lines.append(f"  GPU: {r['gpu_name']}  |  VRAM: {vram}  |  RAM: {r['ram_gb']} GB")
    lines.append(f"  Profile: {r['profile']}  |  Context: {r['ctx_size']:,}  |  Layers: {r['gpu_layers']}  |  Batch: {r['batch_size']}")

    # Recommendations
    recs = r.get("recommendations", {})
    if recs:
        lines.append(f"\n  Model Recommendations:")
        for slot, rec in recs.items():
            lines.append(f"    {slot:8s}: {rec}")

    # Component budget
    lines.append(f"\n  VRAM Budget Analysis:")
    for comp in r["components"]:
        status = "OK" if comp["can_load"] else "BLOCKED"
        pct = f"({comp['pct_of_total']}%)" if comp["pct_of_total"] else ""
        marker = " + " if comp["can_load"] else " X "
        lines.append(f"    {marker} {status:7s}  {comp['name']:30s}  {comp['vram_mb']:>6,} MB  {pct}")

    # Stack analysis
    sa = r["stack_analysis"]
    lines.append(f"\n  Stack Analysis ({sa.get('mode', 'unknown')} mode):")
    if sa.get("mode") == "cpu":
        lines.append(f"    {sa['note']}")
    else:
        status = "FITS" if sa["min_stack_fits"] else "DOES NOT FIT"
        lines.append(f"    Minimum stack (embed+rerank+7B Q4): {sa['min_stack_cost_mb']:,} MB -> {status}")
        if sa["min_stack_fits"]:
            lines.append(f"    Headroom after base: {sa['headroom_after_min_mb']:,} MB ({sa['headroom_after_min_gb']} GB)")

    # Flags
    if r["green_flags"]:
        lines.append(f"\n  What works:")
        for f in r["green_flags"]:
            lines.append(f"    + {f}")
    if r["red_flags"]:
        lines.append(f"\n  Issues / degradation:")
        for f in r["red_flags"]:
            lines.append(f"    ! {f}")

    return "\n".join(lines)


def format_result_markdown(r: dict) -> str:
    """Format a single profile result as markdown."""
    lines = []
    vram = f"{r['vram_total_gb']} GB" if r['vram_total_gb'] else "None"
    lines.append(f"## {r['name'].upper()} — {r['label']}")
    lines.append(f"*{r['real_world']}*\n")
    lines.append(f"| Spec | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| GPU | {r['gpu_name']} |")
    lines.append(f"| VRAM | {vram} |")
    lines.append(f"| RAM | {r['ram_gb']} GB |")
    lines.append(f"| Profile | {r['profile']} |")
    lines.append(f"| Context | {r['ctx_size']:,} tokens |")
    lines.append(f"| GPU Layers | {r['gpu_layers']} |")
    lines.append(f"| Batch Size | {r['batch_size']} |")

    # Recommendations
    recs = r.get("recommendations", {})
    if recs:
        lines.append(f"\n### Model Recommendations")
        for slot, rec in recs.items():
            lines.append(f"- **{slot}**: {rec}")

    # Component budget
    lines.append(f"\n### VRAM Budget")
    lines.append(f"| Component | VRAM | Status | % of Total |")
    lines.append(f"|-----------|------|--------|------------|")
    for comp in r["components"]:
        status = "OK" if comp["can_load"] else "BLOCKED"
        pct = f"{comp['pct_of_total']}%" if comp["pct_of_total"] else "N/A"
        lines.append(f"| {comp['name']} | {comp['vram_mb']:,} MB | {status} | {pct} |")

    # Stack analysis
    sa = r["stack_analysis"]
    lines.append(f"\n### Stack Analysis")
    if sa.get("mode") == "cpu":
        lines.append(f"{sa['note']}")
    else:
        status = "FITS" if sa["min_stack_fits"] else "DOES NOT FIT"
        lines.append(f"- Minimum stack: {sa['min_stack_cost_mb']:,} MB — **{status}**")
        if sa["min_stack_fits"]:
            lines.append(f"- Headroom: {sa['headroom_after_min_mb']:,} MB ({sa['headroom_after_min_gb']} GB)")

    # Flags
    if r["green_flags"]:
        lines.append(f"\n### What Works")
        for f in r["green_flags"]:
            lines.append(f"- {f}")
    if r["red_flags"]:
        lines.append(f"\n### Issues / Degradation")
        for f in r["red_flags"]:
            lines.append(f"- **{f}**")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live API Tests (optional, requires running backend)
# ---------------------------------------------------------------------------

def hit_api(base_url: str, path: str, timeout: int = 10) -> dict:
    """Hit a backend API endpoint."""
    url = f"{base_url.rstrip('/')}{path}"
    start = time.time()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            body = json.loads(resp.read().decode())
            elapsed = round((time.time() - start) * 1000)
            return {"ok": True, "status": resp.status, "ms": elapsed, "body": body}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"ok": False, "status": 0, "ms": elapsed, "error": str(e)}


def live_test(base_url: str) -> list:
    """Hit key endpoints and return results."""
    results = []
    endpoints = [
        ("/api/health", "Health check"),
        ("/api/models", "Model slots"),
        ("/api/domain/info", "Domain info"),
    ]
    for path, desc in endpoints:
        r = hit_api(base_url, path)
        results.append({"path": path, "desc": desc, **r})
    return results


# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------

def comparison_table(results: list) -> str:
    """Build a comparison table across all profiles."""
    lines = []
    lines.append("\n  COMPARISON ACROSS PROFILES")
    lines.append("  " + "-" * 80)

    # Header
    names = [r["name"].upper() for r in results]
    lines.append(f"  {'':30s}  " + "  ".join(f"{n:>10s}" for n in names))
    lines.append(f"  {'':30s}  " + "  ".join(f"{'---':>10s}" for _ in names))

    # VRAM
    vals = [f"{r['vram_total_gb']}G" if r['vram_total_gb'] else "None" for r in results]
    lines.append(f"  {'VRAM':30s}  " + "  ".join(f"{v:>10s}" for v in vals))

    # Profile
    vals = [r["profile"] for r in results]
    lines.append(f"  {'Profile':30s}  " + "  ".join(f"{v:>10s}" for v in vals))

    # Context
    vals = [f"{r['ctx_size']:,}" for r in results]
    lines.append(f"  {'Context (tokens)':30s}  " + "  ".join(f"{v:>10s}" for v in vals))

    # Batch
    vals = [str(r["batch_size"]) for r in results]
    lines.append(f"  {'Batch size':30s}  " + "  ".join(f"{v:>10s}" for v in vals))

    # Components
    lines.append(f"  {'':30s}  " + "  ".join(f"{'---':>10s}" for _ in names))
    if results:
        for i, comp in enumerate(results[0]["components"]):
            comp_name = comp["name"].replace("(", "").replace(")", "").split(" ")[0:3]
            short_name = " ".join(comp_name)
            vals = []
            for r in results:
                c = r["components"][i]
                vals.append("OK" if c["can_load"] else "BLOCKED")
            lines.append(f"  {short_name:30s}  " + "  ".join(f"{v:>10s}" for v in vals))

    # Red flag count
    vals = [str(len(r["red_flags"])) for r in results]
    lines.append(f"  {'':30s}  " + "  ".join(f"{'---':>10s}" for _ in names))
    lines.append(f"  {'Red flags':30s}  " + "  ".join(f"{v:>10s}" for v in vals))

    return "\n".join(lines)


def comparison_table_markdown(results: list) -> str:
    """Build a markdown comparison table."""
    lines = []
    lines.append("## Comparison Table\n")

    names = [r["name"].upper() for r in results]
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|--------|" + "|".join(["-----" for _ in names]) + "|")

    # Key metrics
    row = lambda label, vals: f"| {label} | " + " | ".join(vals) + " |"

    lines.append(row("VRAM", [f"{r['vram_total_gb']}G" if r['vram_total_gb'] else "None" for r in results]))
    lines.append(row("Profile", [r["profile"] for r in results]))
    lines.append(row("Context", [f"{r['ctx_size']:,}" for r in results]))
    lines.append(row("Batch", [str(r["batch_size"]) for r in results]))

    # Components
    if results:
        for i, comp in enumerate(results[0]["components"]):
            short = comp["name"].split("(")[0].strip()
            vals = ["OK" if r["components"][i]["can_load"] else "BLOCKED" for r in results]
            lines.append(row(short, vals))

    lines.append(row("Red Flags", [str(len(r["red_flags"])) for r in results]))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile list display
# ---------------------------------------------------------------------------

def show_profiles():
    """Print profile specs and model recommendations."""
    import services.hardware as hw

    print("\n  ARCA Hardware Profiles")
    print("  " + "-" * 60)
    for name, profile in hw.PROFILES.items():
        recs = hw.MODEL_RECOMMENDATIONS.get(name, {})
        print(f"\n  [{name.upper()}] {profile['description']}")
        print(f"    Context: {profile['ctx_size']:,}  |  GPU layers: {profile['gpu_layers']}  |  Batch: {profile['batch_size']}")
        print(f"    Chat: {recs.get('chat', 'N/A')}")
        print(f"    Vision: {recs.get('vision', 'N/A')}")
        print(f"    Expert: {recs.get('expert', 'N/A')}")

    print(f"\n  Simulation scenarios available:")
    for name, sc in SCENARIOS.items():
        vram = f"{round(sc['vram_mb']/1024, 1)} GB" if sc["vram_mb"] else "None"
        print(f"    {name:12s}  {vram:>8s}  {sc['label']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ARCA Hardware Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/test_hardware_sim.py                     # Test all profiles
  python scripts/test_hardware_sim.py small medium        # Test specific profiles
  python scripts/test_hardware_sim.py --report sim.md     # Save markdown report
  python scripts/test_hardware_sim.py --live              # Also hit backend APIs
  python scripts/test_hardware_sim.py --list              # Show profile info""",
    )
    parser.add_argument("profiles", nargs="*", help="Specific profiles to test (default: all)")
    parser.add_argument("--report", metavar="FILE", help="Save markdown report to file")
    parser.add_argument("--live", action="store_true", help="Also hit running backend APIs")
    parser.add_argument("--host", default="http://localhost:8000", help="Backend URL for --live")
    parser.add_argument("--list", action="store_true", help="Show profile info and exit")
    args = parser.parse_args()

    if args.list:
        show_profiles()
        return

    # Suppress hardware.py log noise
    import logging
    logging.basicConfig(level=logging.CRITICAL)

    # Select profiles
    if args.profiles:
        selected = {k: v for k, v in SCENARIOS.items() if k in args.profiles}
        if not selected:
            print(f"  Unknown profiles: {args.profiles}. Available: {list(SCENARIOS.keys())}")
            sys.exit(1)
    else:
        selected = SCENARIOS

    print(f"\n  ARCA Hardware Simulation Runner")
    print(f"  Testing {len(selected)} hardware configurations...")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run simulations
    results = []
    for name, scenario in selected.items():
        r = simulate_profile(name, scenario)
        results.append(r)
        print(format_result_terminal(r))

    # Comparison
    if len(results) > 1:
        print(comparison_table(results))

    # Live tests
    if args.live:
        print(f"\n  LIVE API TESTS ({args.host})")
        print(f"  " + "-" * 50)
        live_results = live_test(args.host)
        for lr in live_results:
            status = "OK" if lr["ok"] else "FAIL"
            marker = " + " if lr["ok"] else " X "
            print(f"  {marker} {status}  {lr['path']}  ({lr['ms']}ms)  {lr['desc']}")
            if lr.get("body") and lr["ok"]:
                # Show key info from response
                body = lr["body"]
                if "profile" in str(body):
                    print(f"       Profile reported by backend: {body}")

    # Summary
    total_flags = sum(len(r["red_flags"]) for r in results)
    print(f"\n  {'=' * 65}")
    print(f"  SUMMARY: {len(results)} profiles tested, {total_flags} total red flags")
    critical = [r for r in results if not r["stack_analysis"]["min_stack_fits"]]
    if critical:
        print(f"  CRITICAL: {len(critical)} profile(s) cannot run minimum stack: "
              f"{', '.join(r['name'] for r in critical)}")
    print()

    # Save report
    if args.report:
        md_lines = []
        md_lines.append(f"# ARCA Hardware Simulation Report")
        md_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        md_lines.append(f"Profiles tested: {', '.join(r['name'] for r in results)}\n")

        if len(results) > 1:
            md_lines.append(comparison_table_markdown(results))

        for r in results:
            md_lines.append(format_result_markdown(r))

        md_lines.append("---")
        md_lines.append(f"*Report generated by `scripts/test_hardware_sim.py`*")

        report_path = args.report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"  Report saved to: {report_path}")
        print()

    # Cleanup env
    for key in ("ARCA_SIMULATE_VRAM", "ARCA_SIMULATE_GPU", "ARCA_SIMULATE_RAM",
                "ARCA_SIMULATE_CPU", "ARCA_SIMULATE_CORES"):
        os.environ.pop(key, None)


if __name__ == "__main__":
    main()
