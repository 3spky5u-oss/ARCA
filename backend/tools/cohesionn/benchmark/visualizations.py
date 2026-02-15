"""
Visualization Engine
====================
Generate 7 chart types from benchmark results.

Uses matplotlib.Agg backend for headless Docker rendering.
Saves PNGs alongside JSON results.

Charts:
  1. chart_rerankers.png    — Grouped bar: rerankers × metrics
  2. chart_embedders.png    — Grouped bar: embedders × metrics
  3. chart_cross_matrix.png — Seaborn heatmap: embed × rerank
  4. chart_param_sweep.png  — Line charts: value vs quality+latency
  5. chart_llm.png          — Horizontal bar: LLM generation quality
  6. chart_radar.png        — Spider chart: top 3 configs
  7. chart_waterfall.png    — Incremental improvement per optimization
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# Headless rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sns = None
try:
    import seaborn as sns  # type: ignore[no-redef]
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Consistent style
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#795548"]
FIGSIZE = (12, 7)
DPI = 150


def generate_all_charts(
    report_data: Dict[str, Any],
    output_dir: str,
) -> List[str]:
    """Generate all applicable charts. Returns list of created file paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    phases = report_data.get("phases", {})

    if "reranker" in phases:
        path = _chart_grouped_bar(
            phases["reranker"], "Reranker Comparison", output / "chart_rerankers.png"
        )
        if path:
            created.append(path)

    if "embedding" in phases:
        path = _chart_grouped_bar(
            phases["embedding"], "Embedding Model Comparison", output / "chart_embedders.png"
        )
        if path:
            created.append(path)

    if "cross_matrix" in phases:
        path = _chart_cross_matrix(
            phases["cross_matrix"], output / "chart_cross_matrix.png"
        )
        if path:
            created.append(path)

    if "param_sweep" in phases:
        path = _chart_param_sweep(
            phases["param_sweep"], output / "chart_param_sweep.png"
        )
        if path:
            created.append(path)

    if "llm" in phases:
        path = _chart_llm(
            phases["llm"], output / "chart_llm.png"
        )
        if path:
            created.append(path)

    if "ingestion" in phases:
        path = _chart_ingestion_matrix(
            phases["ingestion"], output / "chart_ingestion_matrix.png"
        )
        if path:
            created.append(path)

        path = _chart_chunk_distribution(
            phases["ingestion"], output / "chart_chunk_distribution.png"
        )
        if path:
            created.append(path)

    if "ablation" in phases:
        path = _chart_ablation(
            phases["ablation"], output / "chart_ablation.png"
        )
        if path:
            created.append(path)

    # Radar and waterfall need results from multiple phases
    if len(phases) >= 2:
        path = _chart_radar(report_data, output / "chart_radar.png")
        if path:
            created.append(path)

        path = _chart_waterfall(report_data, output / "chart_waterfall.png")
        if path:
            created.append(path)

        path = _chart_tier_breakdown(report_data, output / "chart_tier_breakdown.png")
        if path:
            created.append(path)

    if len(phases) >= 3:
        path = _chart_summary_dashboard(report_data, output / "chart_summary_dashboard.png")
        if path:
            created.append(path)

    return created


# ── Chart 1 & 2: Grouped Bar ────────────────────────────────────────────────

def _chart_grouped_bar(
    phase_data: Dict[str, Any],
    title: str,
    output_path: Path,
) -> Optional[str]:
    """Grouped bar chart comparing variants across metrics."""
    variants = phase_data.get("variants", [])
    valid = [v for v in variants if v.get("aggregate") and not v.get("error")]
    if not valid:
        return None

    metrics = ["avg_keyword_hits", "avg_mrr", "avg_ndcg", "avg_diversity", "avg_composite"]
    metric_labels = ["Keywords", "MRR", "nDCG", "Diversity", "Composite"]
    names = [v["variant_name"] for v in valid]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(names))
    width = 0.8 / len(metrics)

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [v["aggregate"].get(metric, 0) for v in valid]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            values,
            width,
            label=label,
            color=COLORS[i % len(COLORS)],
            alpha=0.85,
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set_xlabel("Model Variant")
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Add latency as secondary axis
    ax2 = ax.twinx()
    latencies = [v["aggregate"].get("avg_latency_ms", 0) for v in valid]
    ax2.plot(x, latencies, "k--o", alpha=0.5, markersize=5, label="Latency (ms)")
    ax2.set_ylabel("Latency (ms)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


# ── Chart 3: Cross-Matrix Heatmap ───────────────────────────────────────────

def _chart_cross_matrix(
    phase_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Seaborn heatmap of embedder × reranker composite scores."""
    variants = phase_data.get("variants", [])
    if not variants:
        return None

    # Find heatmap_data in metadata
    heatmap_data = None
    for v in variants:
        hd = v.get("metadata", {}).get("heatmap_data")
        if hd:
            heatmap_data = hd
            break

    if not heatmap_data:
        return None

    embedders = list(heatmap_data.keys())
    rerankers = list(set(
        rk for rks in heatmap_data.values() for rk in rks.keys()
    ))
    rerankers.sort()

    matrix = []
    for emb in embedders:
        row = [heatmap_data.get(emb, {}).get(rer, 0) for rer in rerankers]
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

    if HAS_SEABORN and sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            xticklabels=rerankers,
            yticklabels=embedders,
            cmap="YlOrRd",
            ax=ax,
            vmin=0,
            vmax=1,
            linewidths=0.5,
        )
    else:
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(rerankers)))
        ax.set_xticklabels(rerankers, rotation=45, ha="right")
        ax.set_yticks(range(len(embedders)))
        ax.set_yticklabels(embedders)
        for i in range(len(embedders)):
            for j in range(len(rerankers)):
                ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)

    ax.set_title("Cross-Matrix: Embedder × Reranker Composite Score", fontweight="bold")
    ax.set_xlabel("Reranker")
    ax.set_ylabel("Embedder")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


# ── Chart 4: Parameter Sweep ────────────────────────────────────────────────

def _chart_param_sweep(
    phase_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Line charts showing parameter value vs quality and latency."""
    variants = phase_data.get("variants", [])
    valid = [v for v in variants if v.get("aggregate") and v.get("metadata")]
    if not valid:
        return None

    # Group by parameter
    params: Dict[str, List] = {}
    for v in valid:
        param = v["metadata"].get("param", "unknown")
        params.setdefault(param, []).append(v)

    n_params = len(params)
    if n_params == 0:
        return None

    cols = min(3, n_params)
    rows = math.ceil(n_params / cols)
    fig, axes_raw = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=DPI)
    # Normalize axes to a flat list
    if n_params == 1:
        axes_list = [axes_raw]
    elif hasattr(axes_raw, "flatten"):
        axes_list = list(axes_raw.flatten())
    else:
        axes_list = [axes_raw]

    for idx, (param_name, param_variants) in enumerate(sorted(params.items())):
        if idx >= len(axes_list):
            break
        ax = axes_list[idx]

        # Sort by parameter value
        param_variants.sort(key=lambda v: v["metadata"].get("value", 0))
        x_vals = [v["metadata"].get("value", 0) for v in param_variants]
        composites = [v["aggregate"].get("avg_composite", 0) for v in param_variants]
        keywords = [v["aggregate"].get("avg_keyword_hits", 0) for v in param_variants]
        latencies = [v["aggregate"].get("avg_latency_ms", 0) for v in param_variants]

        ax.plot(x_vals, composites, "o-", color=COLORS[0], label="Composite", linewidth=2)
        ax.plot(x_vals, keywords, "s--", color=COLORS[1], label="Keywords", linewidth=1.5)
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(param_name, fontweight="bold", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

        # Secondary latency axis
        ax2 = ax.twinx()
        ax2.plot(x_vals, latencies, "k:^", alpha=0.4, markersize=4, label="Latency")
        ax2.set_ylabel("ms", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

    # Hide empty subplots
    for idx in range(n_params, len(axes_list)):
        axes_list[idx].set_visible(False)

    fig.suptitle("Parameter Sweep Results", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return str(output_path)


# ── Chart 5: LLM Comparison ─────────────────────────────────────────────────

def _chart_llm(
    phase_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Horizontal bar chart comparing LLM generation quality."""
    variants = phase_data.get("variants", [])
    valid = [v for v in variants if v.get("aggregate") and not v.get("error")]
    if not valid:
        return None

    valid.sort(key=lambda v: v["aggregate"].get("avg_composite", 0))
    names = [v["variant_name"] for v in valid]
    composites = [v["aggregate"]["avg_composite"] for v in valid]
    keywords = [v["aggregate"]["avg_keyword_hits"] for v in valid]

    fig, ax = plt.subplots(figsize=(10, max(4, len(valid) * 1.2)), dpi=DPI)
    y = np.arange(len(names))

    bars = ax.barh(y - 0.15, composites, 0.3, label="Composite", color=COLORS[0], alpha=0.85)
    ax.barh(y + 0.15, keywords, 0.3, label="Keywords", color=COLORS[1], alpha=0.85)

    for bar, val in zip(bars, composites):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_title("LLM Generation Quality Comparison", fontweight="bold", fontsize=14)
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


# ── Chart 6: Radar / Spider ─────────────────────────────────────────────────

def _chart_radar(
    report_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Spider chart comparing top 3 configs across all metrics."""
    ranking = report_data.get("overall_ranking", [])
    if len(ranking) < 2:
        return None

    # Collect top 3 unique variants with full data
    top_variants = []
    seen = set()
    for r in ranking:
        key = f"{r['phase']}:{r['variant']}"
        if key not in seen:
            seen.add(key)
            # Find full variant data
            phase_data = report_data["phases"].get(r["phase"], {})
            for v in phase_data.get("variants", []):
                if v["variant_name"] == r["variant"] and v.get("aggregate"):
                    top_variants.append(v)
                    break
        if len(top_variants) >= 3:
            break

    if len(top_variants) < 2:
        return None

    categories = ["Keywords", "Entities", "MRR", "nDCG", "Diversity"]
    metric_keys = ["avg_keyword_hits", "avg_entity_hits", "avg_mrr", "avg_ndcg", "avg_diversity"]

    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI, subplot_kw=dict(polar=True))

    for i, v in enumerate(top_variants):
        agg = v["aggregate"]
        values = [agg.get(k, 0) for k in metric_keys]
        values += values[:1]
        ax.plot(angles, values, "o-", color=COLORS[i], linewidth=2, label=v["variant_name"])
        ax.fill(angles, values, color=COLORS[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Top Configurations — Metric Radar", fontweight="bold", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return str(output_path)


# ── Chart 7: Waterfall ──────────────────────────────────────────────────────

def _chart_waterfall(
    report_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Waterfall chart showing incremental improvement per phase."""
    phases = report_data.get("phases", {})
    if len(phases) < 2:
        return None

    # Get phase winners in order
    phase_order = ["reranker", "embedding", "cross_matrix", "param_sweep", "llm"]
    steps = []
    baseline = 0.0

    for phase_name in phase_order:
        if phase_name not in phases:
            continue
        phase = phases[phase_name]
        ranking = phase.get("ranking", [])
        if ranking:
            score = ranking[0]["composite"]
            delta = score - baseline if baseline > 0 else score
            steps.append({
                "label": f"{phase_name}\n({ranking[0]['variant']})",
                "score": score,
                "delta": delta,
            })
            baseline = score

    if len(steps) < 2:
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(steps))
    bottoms = []
    heights = []

    prev = 0
    for s in steps:
        bottoms.append(prev)
        heights.append(s["delta"])
        prev = s["score"]

    colors_list = [COLORS[1] if h >= 0 else COLORS[3] for h in heights]
    colors_list[0] = COLORS[0]  # First bar is baseline

    ax.bar(x, heights, bottom=bottoms, color=colors_list, alpha=0.85, edgecolor="white")

    # Labels
    for i, s in enumerate(steps):
        ax.text(
            i, s["score"] + 0.01,
            f"{s['score']:.3f}",
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )
        if i > 0:
            sign = "+" if s["delta"] >= 0 else ""
            ax.text(
                i, bottoms[i] + heights[i] / 2,
                f"{sign}{s['delta']:.3f}",
                ha="center", va="center", fontsize=8, color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in steps], fontsize=9)
    ax.set_ylabel("Composite Score")
    ax.set_title("Incremental Improvement per Optimization Phase", fontweight="bold", fontsize=14)
    ax.set_ylim(0, max(s["score"] for s in steps) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


# ── Chart 8: Ingestion Matrix Heatmap ──────────────────────────────────────

def _chart_ingestion_matrix(
    phase_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Heatmap of extractor × chunk_size → composite score."""
    variants = phase_data.get("variants", [])
    valid = [v for v in variants if v.get("aggregate") and not v.get("error")]
    if not valid:
        return None

    # Build matrix data: group by (extractor, chunk_size)
    extractors = sorted(set(v.get("metadata", {}).get("extractor", "?") for v in valid))
    chunk_sizes = sorted(set(v.get("metadata", {}).get("chunk_size", 0) for v in valid))

    if len(extractors) < 1 or len(chunk_sizes) < 2:
        return _chart_grouped_bar(phase_data, "Ingestion Shootout", output_path)

    matrix = []
    for ext in extractors:
        row = []
        for cs in chunk_sizes:
            score = 0.0
            for v in valid:
                m = v.get("metadata", {})
                if m.get("extractor") == ext and m.get("chunk_size") == cs:
                    score = v["aggregate"].get("avg_composite", 0)
                    break
            row.append(score)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, max(4, len(extractors) * 1.5)), dpi=DPI)
    cs_labels = [str(cs) for cs in chunk_sizes]

    if HAS_SEABORN and sns is not None:
        sns.heatmap(
            matrix, annot=True, fmt=".3f",
            xticklabels=cs_labels, yticklabels=extractors,
            cmap="YlOrRd", ax=ax, vmin=0, vmax=1, linewidths=0.5,
        )
    else:
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cs_labels)))
        ax.set_xticklabels(cs_labels)
        ax.set_yticks(range(len(extractors)))
        ax.set_yticklabels(extractors)
        for i in range(len(extractors)):
            for j in range(len(cs_labels)):
                ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)

    ax.set_title("Ingestion: Extractor × Chunk Size → Composite", fontweight="bold")
    ax.set_xlabel("Chunk Size")
    ax.set_ylabel("Extractor")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


# ── Chart 9: Chunk Distribution ────────────────────────────────────────────

def _chart_chunk_distribution(
    phase_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Bar chart of chunk counts per ingestion variant."""
    variants = phase_data.get("variants", [])
    valid = [v for v in variants if v.get("metadata") and not v.get("error")]
    if not valid:
        return None

    names = [v["variant_name"] for v in valid]
    chunks = [v["metadata"].get("chunks_created", 0) for v in valid]
    composites = [v.get("aggregate", {}).get("avg_composite", 0) for v in valid]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x = np.arange(len(names))

    bars = ax.bar(x, chunks, color=COLORS[0], alpha=0.85)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Chunks Created")
    ax.set_title("Chunk Count per Ingestion Variant", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, chunks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=7)

    ax2 = ax.twinx()
    ax2.plot(x, composites, "ro-", alpha=0.7, markersize=6, label="Composite")
    ax2.set_ylabel("Composite Score", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


# ── Chart 10: Ablation Delta Bars ──────────────────────────────────────────

def _chart_ablation(
    phase_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Horizontal bar chart of component toggle deltas."""
    variants = phase_data.get("variants", [])
    valid = [v for v in variants if v.get("metadata") and not v.get("error")]
    if not valid:
        return None

    valid.sort(key=lambda v: abs(v["metadata"].get("delta", 0)), reverse=True)
    labels = [v["metadata"].get("label", v["variant_name"]) for v in valid]
    deltas = [v["metadata"].get("delta", 0) for v in valid]

    fig, ax = plt.subplots(figsize=(10, max(4, len(valid) * 0.8)), dpi=DPI)
    y = np.arange(len(labels))

    bar_colors = [COLORS[1] if d >= 0 else COLORS[3] for d in deltas]
    bars = ax.barh(y, deltas, color=bar_colors, alpha=0.85, edgecolor="white")

    for bar, delta in zip(bars, deltas):
        sign = "+" if delta >= 0 else ""
        x_pos = bar.get_width() + 0.002 if delta >= 0 else bar.get_width() - 0.002
        ha = "left" if delta >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{sign}{delta:.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Composite Score Delta (ON - OFF)")
    ax.set_title("Pipeline Component Ablation", fontweight="bold", fontsize=14)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


# ── Chart 11: Tier Breakdown ──────────────────────────────────────────────

def _chart_tier_breakdown(
    report_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """Grouped bar chart of per-tier scores for top 3 configs."""
    ranking = report_data.get("overall_ranking", [])
    phases = report_data.get("phases", {})
    if len(ranking) < 2:
        return None

    configs = []
    seen = set()
    for r in ranking:
        key = f"{r['phase']}:{r['variant']}"
        if key in seen:
            continue
        seen.add(key)
        phase_data = phases.get(r["phase"], {})
        for v in phase_data.get("variants", []):
            if v["variant_name"] == r["variant"] and v.get("aggregate"):
                by_tier = v["aggregate"].get("by_tier", {})
                if by_tier:
                    configs.append((r["variant"], by_tier))
                    break
        if len(configs) >= 3:
            break

    if len(configs) < 2:
        return None

    tiers = sorted(set(tier for _, bt in configs for tier in bt))
    n_configs = len(configs)

    fig, ax = plt.subplots(figsize=(max(10, len(tiers) * 2), 7), dpi=DPI)
    x = np.arange(len(tiers))
    width = 0.8 / n_configs

    for i, (name, by_tier) in enumerate(configs):
        values = [by_tier.get(t, {}).get("avg_composite", 0) for t in tiers]
        ax.bar(x + i * width - 0.4 + width / 2, values, width,
               label=name, color=COLORS[i % len(COLORS)], alpha=0.85)

    ax.set_xlabel("Query Tier")
    ax.set_ylabel("Composite Score")
    ax.set_title("Per-Tier Scores: Top Configurations", fontweight="bold", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=30, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


# ── Chart 12: Summary Dashboard ───────────────────────────────────────────

def _chart_summary_dashboard(
    report_data: Dict[str, Any],
    output_path: Path,
) -> Optional[str]:
    """2×3 grid summarizing key findings from each phase."""
    phases = report_data.get("phases", {})
    available: List[Optional[str]] = [
        p for p in ["ingestion", "reranker", "embedding", "param_sweep", "llm", "ablation"]
        if p in phases
    ]
    if len(available) < 3:
        return None

    while len(available) < 6:
        available.append(None)

    fig, axes_raw = plt.subplots(2, 3, figsize=(18, 10), dpi=DPI)
    axes_list = list(axes_raw.flatten())

    for idx, phase_name in enumerate(available[:6]):
        ax = axes_list[idx]
        if phase_name is None:
            ax.set_visible(False)
            continue

        phase = phases[phase_name]
        ranking = phase.get("ranking", [])[:5]
        if not ranking:
            ax.set_visible(False)
            continue

        names = [r["variant"][:15] for r in ranking]
        scores = [r["composite"] for r in ranking]

        ax.barh(range(len(names)), scores,
                color=[COLORS[0]] + [COLORS[5]] * (len(names) - 1), alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_title(phase_name.replace("_", " ").title(), fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        if scores:
            ax.text(scores[0] + 0.01, 0, f"{scores[0]:.3f}",
                    va="center", fontsize=9, fontweight="bold", color=COLORS[0])

    fig.suptitle("ARCA RAG Pipeline Benchmark — Summary", fontweight="bold", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return str(output_path)
