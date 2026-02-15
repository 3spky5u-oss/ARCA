"""
Report Generator
================
JSON output + ANSI console summary for benchmark results.

Incremental saves after each phase (crash-safe).
Compatible with existing benchmark_compare.py output format.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── ANSI Colors ──────────────────────────────────────────────────────────────

class Colors:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RESET = "\033[0m"
    UNDERLINE = "\033[4m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


# ── Report Data Structure ────────────────────────────────────────────────────


class ShootoutReport:
    """Manages benchmark results, incremental saves, and console output."""

    def __init__(self, output_dir: str, config_dict: Optional[Dict] = None):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"shootout_{ts}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config_dict or {},
            "phases": {},
            "overall_winner": None,
            "overall_ranking": [],
        }

    @property
    def json_path(self) -> Path:
        return self.output_dir / "results.json"

    def add_phase(
        self,
        phase_name: str,
        results: List[Dict[str, Any]],
        duration_s: float,
    ):
        """Add phase results and save incrementally."""
        valid = [r for r in results if r.get("aggregate") and not r.get("error")]

        if phase_name == "ablation":
            # Rank by absolute delta (impact), not by composite
            valid.sort(
                key=lambda r: abs(r.get("metadata", {}).get("delta", 0)),
                reverse=True,
            )
            ranking = [
                {
                    "rank": i + 1,
                    "variant": r["variant_name"],
                    "composite": r["aggregate"]["avg_composite"],
                    "delta": r.get("metadata", {}).get("delta", 0),
                }
                for i, r in enumerate(valid)
            ]
        else:
            # Normal ranking by composite score
            valid.sort(
                key=lambda r: r["aggregate"].get("avg_composite", 0),
                reverse=True,
            )
            ranking = [
                {
                    "rank": i + 1,
                    "variant": r["variant_name"],
                    "composite": r["aggregate"]["avg_composite"],
                }
                for i, r in enumerate(valid)
            ]

        self.data["phases"][phase_name] = {
            "duration_s": round(duration_s, 1),
            "n_variants": len(results),
            "winner": ranking[0]["variant"] if ranking else None,
            "ranking": ranking,
            "variants": results,
        }

        self._save()

    def finalize(self):
        """Compute overall winner across all phases and save."""
        all_composites: List[Dict[str, float]] = []
        for phase_name, phase_data in self.data["phases"].items():
            if phase_name == "ablation":
                continue  # Ablation deltas don't compete with composite scores
            for r in phase_data.get("ranking", []):
                all_composites.append({
                    "phase": phase_name,
                    "variant": r["variant"],
                    "composite": r["composite"],
                })

        all_composites.sort(key=lambda x: x["composite"], reverse=True)
        self.data["overall_ranking"] = all_composites[:10]
        if all_composites:
            self.data["overall_winner"] = all_composites[0]

        self._save()

    def _save(self):
        """Write JSON to disk."""
        self.json_path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, default=str)
        )

    # ── Console Output ───────────────────────────────────────────────────────

    def print_phase_summary(self, phase_name: str):
        """Print a concise phase summary to console."""
        phase = self.data["phases"].get(phase_name, {})
        if not phase:
            return

        ranking = phase.get("ranking", [])
        winner = phase.get("winner", "N/A")
        n = phase.get("n_variants", 0)
        duration = phase.get("duration_s", 0)

        print()
        print(_c(f"  ═══ Phase: {phase_name} ═══", Colors.BOLD))
        print(f"  {n} variants tested in {duration:.0f}s")
        print(f"  Winner: {_c(winner, Colors.GREEN + Colors.BOLD)}")
        print()

        # Table header
        print(f"  {'Rank':>4}  {'Variant':<30}  {'Composite':>9}  {'Keywords':>8}  "
              f"{'MRR':>6}  {'nDCG':>6}  {'Latency':>8}")
        print(f"  {'─' * 82}")

        variants = {v["variant_name"]: v for v in phase.get("variants", [])}
        for r in ranking[:10]:
            v = variants.get(r["variant"], {})
            agg = v.get("aggregate", {})

            # Color code by rank
            rank = r["rank"]
            if rank == 1:
                color = Colors.GREEN
            elif rank <= 3:
                color = Colors.YELLOW
            else:
                color = Colors.DIM

            rank_str = f"#{rank:<3}"
            comp_str = f"{r['composite']:>8.3f}"
            print(
                f"  {_c(rank_str, color)}  "
                f"{r['variant']:<30}  "
                f"{_c(comp_str, color)}  "
                f"{agg.get('avg_keyword_hits', 0):>7.1%}  "
                f"{agg.get('avg_mrr', 0):>5.3f}  "
                f"{agg.get('avg_ndcg', 0):>5.3f}  "
                f"{agg.get('avg_latency_ms', 0):>6.0f}ms"
            )

        # Tier breakdown for winner
        if ranking:
            winner_v = variants.get(ranking[0]["variant"], {})
            by_tier = winner_v.get("aggregate", {}).get("by_tier", {})
            if by_tier:
                print()
                print(f"  {_c('Tier breakdown (winner):', Colors.CYAN)}")
                for tier, tier_data in sorted(by_tier.items()):
                    bar_len = int(tier_data.get("avg_composite", 0) * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    print(
                        f"    {tier:<18} {bar}  "
                        f"comp={tier_data.get('avg_composite', 0):.3f}  "
                        f"kw={tier_data.get('avg_keyword_hits', 0):.0%}  "
                        f"n={tier_data.get('n_queries', 0)}"
                    )

        # Errors
        error_variants = [v for v in phase.get("variants", []) if v.get("error")]
        if error_variants:
            print()
            print(f"  {_c('Errors:', Colors.RED)}")
            for v in error_variants:
                print(f"    {v['variant_name']}: {v['error']}")

    def print_final_summary(self):
        """Print overall summary across all phases."""
        print()
        print(_c("  ════════════════════════════════════════════════", Colors.BOLD))
        print(_c("  SHOOTOUT COMPLETE", Colors.BOLD + Colors.MAGENTA))
        print(_c("  ════════════════════════════════════════════════", Colors.BOLD))

        for phase_name in self.data["phases"]:
            phase = self.data["phases"][phase_name]
            winner = phase.get("winner", "N/A")
            score = 0.0
            for r in phase.get("ranking", []):
                if r["variant"] == winner:
                    score = r["composite"]
                    break
            print(f"  {phase_name:<20} → {_c(winner, Colors.GREEN)} ({score:.3f})")

        overall = self.data.get("overall_winner")
        if overall:
            print()
            print(
                f"  Overall best: {_c(overall['variant'], Colors.GREEN + Colors.BOLD)} "
                f"({overall['phase']}) — composite {overall['composite']:.3f}"
            )

        print(f"\n  Results: {_c(str(self.json_path), Colors.UNDERLINE)}")
        print(f"  Charts:  {_c(str(self.output_dir), Colors.UNDERLINE)}")
