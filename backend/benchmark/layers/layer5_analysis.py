"""
Layer 5: Statistical Analysis + Visualization

Reads all layer outputs and produces:
- Heatmap: chunk_size x chunk_overlap -> composite score
- Bar chart: marginal value of each retrieval toggle
- Scatter: latency vs quality per retrieval config
- Per-tier breakdown tables
- Markdown report for presentation
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from benchmark.layers.base import BaseLayer, LayerResult

logger = logging.getLogger(__name__)


class AnalysisLayer(BaseLayer):
    """Layer 5: Statistical analysis and visualization."""

    LAYER_NAME = "layer5_analysis"

    def execute(self, result: LayerResult) -> LayerResult:
        # Load all layer results
        l0_summary = self._load_json("layer0_chunking", "summary.json")
        l1_summary = self._load_json("layer1_retrieval", "summary.json")
        l2_summary = self._load_json("layer2_params", "summary.json")
        l4_summary = self._load_json("layer4_judge", "summary.json")
        l4_judgments = self._load_json("layer4_judge", "judgments.json")

        # Generate visualizations
        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        charts_generated = []

        if l0_summary:
            try:
                self._heatmap_chunking(l0_summary, charts_dir)
                charts_generated.append("chunking_heatmap.png")
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")
                result.errors.append(f"Heatmap: {e}")

        if l1_summary:
            try:
                self._bar_retrieval(l1_summary, charts_dir)
                charts_generated.append("retrieval_comparison.png")
            except Exception as e:
                logger.warning(f"Retrieval bar chart failed: {e}")
                result.errors.append(f"Bar chart: {e}")

            try:
                self._scatter_latency_quality(l1_summary, charts_dir)
                charts_generated.append("latency_vs_quality.png")
            except Exception as e:
                logger.warning(f"Scatter plot failed: {e}")
                result.errors.append(f"Scatter: {e}")

        if l2_summary:
            try:
                self._param_sensitivity(l2_summary, charts_dir)
                charts_generated.append("param_sensitivity.png")
            except Exception as e:
                logger.warning(f"Param sensitivity chart failed: {e}")
                result.errors.append(f"Param sensitivity: {e}")

        # Generate markdown report
        report = self._generate_report(l0_summary, l1_summary, l2_summary, l4_summary, l4_judgments)
        report_path = self.output_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")

        result.summary = {
            "charts_generated": charts_generated,
            "report_path": str(report_path),
        }
        result.configs_completed = len(charts_generated) + 1  # Charts + report

        return result

    def _load_json(self, layer: str, filename: str):
        """Load a JSON file from a layer's output directory."""
        path = Path(self.config.output_dir) / layer / filename
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _heatmap_chunking(self, l0_summary: Dict, charts_dir: Path):
        """Generate chunk_size x overlap heatmap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        ranking = l0_summary.get("ranking", [])
        if not ranking:
            return

        # Extract unique sizes and overlaps
        sizes = sorted(set(r["chunk_size"] for r in ranking))
        overlaps = sorted(set(r["chunk_overlap"] for r in ranking))

        # Build score matrix (context_prefix=True only for cleaner heatmap)
        prefix_scores = {}
        for r in ranking:
            key = (r["chunk_size"], r["chunk_overlap"])
            if r.get("context_prefix", True):
                prefix_scores[key] = r["composite"]

        matrix = np.zeros((len(sizes), len(overlaps)))
        for i, size in enumerate(sizes):
            for j, overlap in enumerate(overlaps):
                matrix[i, j] = prefix_scores.get((size, overlap), 0)

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(overlaps)))
        ax.set_xticklabels(overlaps)
        ax.set_yticks(range(len(sizes)))
        ax.set_yticklabels(sizes)
        ax.set_xlabel("Chunk Overlap (chars)")
        ax.set_ylabel("Chunk Size (chars)")
        ax.set_title("Chunking Sweep: Composite Score Heatmap")

        # Add text annotations
        for i in range(len(sizes)):
            for j in range(len(overlaps)):
                if matrix[i, j] > 0:
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)

        plt.colorbar(im, label="Composite Score")
        plt.tight_layout()
        plt.savefig(charts_dir / "chunking_heatmap.png", dpi=150)
        plt.close()
        logger.info("Generated chunking heatmap")

    def _bar_retrieval(self, l1_summary: Dict, charts_dir: Path):
        """Generate retrieval config comparison bar chart."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ranking = l1_summary.get("ranking", [])
        if not ranking:
            return

        names = [r["config_name"] for r in ranking]
        scores = [r["composite"] for r in ranking]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(names)), scores, color="steelblue")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Composite Score")
        ax.set_title("Retrieval Config Comparison")
        ax.invert_yaxis()

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{score:.4f}", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(charts_dir / "retrieval_comparison.png", dpi=150)
        plt.close()
        logger.info("Generated retrieval comparison chart")

    def _scatter_latency_quality(self, l1_summary: Dict, charts_dir: Path):
        """Generate latency vs quality scatter plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ranking = l1_summary.get("ranking", [])
        if not ranking:
            return

        names = [r["config_name"] for r in ranking]
        latencies = [r.get("latency_ms", 0) for r in ranking]
        scores = [r["composite"] for r in ranking]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(latencies, scores, s=80, c="steelblue", edgecolors="navy", alpha=0.8)

        for i, name in enumerate(names):
            ax.annotate(name, (latencies[i], scores[i]), fontsize=7,
                       xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("Average Latency (ms)")
        ax.set_ylabel("Composite Score")
        ax.set_title("Retrieval Quality vs Latency")

        plt.tight_layout()
        plt.savefig(charts_dir / "latency_vs_quality.png", dpi=150)
        plt.close()
        logger.info("Generated latency vs quality scatter")

    def _param_sensitivity(self, l2_summary: Dict, charts_dir: Path):
        """Generate parameter sensitivity charts."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sweeps = l2_summary.get("param_sweeps", {})
        if not sweeps:
            return

        n_params = len(sweeps)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (param_name, data) in enumerate(sweeps.items()):
            if idx >= 6:
                break

            ax = axes[idx]
            values = [v["value"] for v in data["all_values"]]
            scores = [v["composite"] for v in data["all_values"]]

            ax.plot(values, scores, "o-", color="steelblue", markersize=8)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Composite")
            ax.set_title(f"{param_name}")

            # Highlight best
            best_idx = scores.index(max(scores))
            ax.plot(values[best_idx], scores[best_idx], "r*", markersize=15)

        # Hide unused subplots
        for idx in range(len(sweeps), 6):
            axes[idx].set_visible(False)

        plt.suptitle("Parameter Sensitivity Analysis", fontsize=14)
        plt.tight_layout()
        plt.savefig(charts_dir / "param_sensitivity.png", dpi=150)
        plt.close()
        logger.info("Generated parameter sensitivity chart")

    def _generate_report(self, l0, l1, l2, l4, l4_judgments) -> str:
        """Generate comprehensive markdown report."""
        lines = [
            "# ARCA Benchmark Report",
            "",
            f"**Run ID:** {self.config.run_id}",
            f"**Corpus:** {self.config.corpus_dir}",
            "",
        ]

        # Layer 0 results
        if l0:
            lines.extend([
                "## Layer 0: Chunking Sweep",
                "",
                f"Tested {l0.get('total_configs', 0)} configurations.",
                "",
                "### Top 10 Configurations",
                "",
                "| Rank | Config | Size | Overlap | Prefix | Composite |",
                "|------|--------|------|---------|--------|-----------|",
            ])
            for r in l0.get("ranking", [])[:10]:
                lines.append(
                    f"| {r['rank']} | {r['config_id']} | {r['chunk_size']} | "
                    f"{r['chunk_overlap']} | {r.get('context_prefix', '-')} | "
                    f"{r['composite']:.4f} |"
                )
            lines.append("")

        # Layer 1 results
        if l1:
            lines.extend([
                "## Layer 1: Retrieval Configuration Sweep",
                "",
                "| Rank | Config | Composite | Latency (ms) |",
                "|------|--------|-----------|--------------|",
            ])
            for r in l1.get("ranking", []):
                lines.append(
                    f"| {r['rank']} | {r['config_name']} | "
                    f"{r['composite']:.4f} | {r.get('latency_ms', 0):.0f} |"
                )
            lines.append("")

        # Layer 2 results
        if l2:
            lines.extend([
                "## Layer 2: Parameter Optimization",
                "",
                "| Parameter | Best Value | Composite |",
                "|-----------|------------|-----------|",
            ])
            for param, data in l2.get("param_sweeps", {}).items():
                lines.append(
                    f"| {param} | {data['best_value']} | {data['best_composite']:.4f} |"
                )
            lines.append("")

        # Layer 4 results
        if l4:
            lines.extend([
                "## Layer 4: LLM-as-Judge Evaluation",
                "",
                f"- Average Relevance: {l4.get('avg_relevance', 0):.2f}/5",
                f"- Average Accuracy: {l4.get('avg_accuracy', 0):.2f}/5",
                f"- Average Completeness: {l4.get('avg_completeness', 0):.2f}/5",
                f"- Overall Average: {l4.get('avg_overall', 0):.2f}/5",
                "",
            ])

            if l4.get("by_tier"):
                lines.extend([
                    "### Per-Tier Breakdown",
                    "",
                    "| Tier | Count | Relevance | Accuracy | Completeness | Overall |",
                    "|------|-------|-----------|----------|--------------|---------|",
                ])
                for tier, data in l4["by_tier"].items():
                    lines.append(
                        f"| {tier} | {data['count']} | {data['avg_relevance']:.2f} | "
                        f"{data['avg_accuracy']:.2f} | {data['avg_completeness']:.2f} | "
                        f"{data['avg_overall']:.2f} |"
                    )
                lines.append("")

        lines.extend([
            "---",
            "*Generated by ARCA Benchmark Harness v2*",
        ])

        return "\n".join(lines)
