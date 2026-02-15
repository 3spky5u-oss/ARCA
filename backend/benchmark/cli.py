"""
Benchmark Harness v2 CLI

Usage (inside Docker container):
    python -m benchmark.cli layer0 --corpus "/app/data/Synthetic Reports/"
    python -m benchmark.cli layer1 --run-id 20260212_031500
    python -m benchmark.cli layer2 --run-id 20260212_031500
    python -m benchmark.cli layer3 --run-id 20260212_031500
    python -m benchmark.cli layer4 --run-id 20260212_031500
    python -m benchmark.cli layer5 --run-id 20260212_031500
    python -m benchmark.cli layer6 --run-id 20260212_031500
    python -m benchmark.cli full --corpus "/app/data/Synthetic Reports/"
    python -m benchmark.cli status --run-id 20260212_031500
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def _build_config(args) -> "BenchmarkConfig":
    """Build BenchmarkConfig from CLI args."""
    from benchmark.config import BenchmarkConfig

    kwargs = {}
    if hasattr(args, "run_id") and args.run_id:
        kwargs["run_id"] = args.run_id
    if hasattr(args, "corpus") and args.corpus:
        kwargs["corpus_dir"] = args.corpus
    if hasattr(args, "max_configs") and args.max_configs:
        kwargs["max_configs"] = args.max_configs
    if hasattr(args, "topic") and args.topic:
        kwargs["topic"] = args.topic

    return BenchmarkConfig(**kwargs)


def _get_checkpoint(config) -> "CheckpointManager":
    """Create checkpoint manager for a run."""
    from benchmark.checkpoint import CheckpointManager

    return CheckpointManager(Path(config.output_dir) / "checkpoints")


def cmd_layer0(args):
    """Run Layer 0: Chunking sweep."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer0_chunking import ChunkingSweepLayer

    layer = ChunkingSweepLayer(config, checkpoint)
    result = layer.run()

    # Print query source if available
    query_meta_path = Path(config.output_dir) / "layer0_chunking" / "query_meta.json"
    query_source = "unknown"
    query_count = 0
    if query_meta_path.exists():
        try:
            meta = json.loads(query_meta_path.read_text(encoding="utf-8"))
            query_source = meta.get("source", "unknown")
            query_count = meta.get("count", 0)
        except Exception:
            pass

    print(f"\n{'=' * 60}")
    print(f"Layer 0 Complete: {result.status}")
    print(f"  Query source: {query_source} ({query_count} queries)")
    print(f"  Configs tested: {result.configs_completed}/{result.configs_total}")
    print(f"  Skipped (checkpoint): {result.configs_skipped}")
    if result.best_config_id:
        print(f"  Best: {result.best_config_id} (composite={result.best_score:.4f})")
    print(f"  Duration: {result.duration_seconds:.0f}s")
    print(f"  Output: {config.output_dir}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
    print(f"{'=' * 60}")


def cmd_layer1(args):
    """Run Layer 1: Retrieval sweep."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer1_retrieval import RetrievalSweepLayer

    layer = RetrievalSweepLayer(config, checkpoint)
    result = layer.run()

    print(f"\nLayer 1 Complete: {result.status}")
    if result.best_config_id:
        print(
            f"  Best retrieval config: {result.best_config_id} "
            f"({result.best_score:.4f})"
        )
    print(f"  Duration: {result.duration_seconds:.0f}s")


def cmd_layer2(args):
    """Run Layer 2: Parameter sweep."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer2_params import ParamSweepLayer

    layer = ParamSweepLayer(config, checkpoint)
    result = layer.run()

    print(f"\nLayer 2 Complete: {result.status}")
    print(f"  Params tested: {result.configs_completed}/{result.configs_total}")
    print(f"  Duration: {result.duration_seconds:.0f}s")


def cmd_layer3(args):
    """Run Layer 3: Answer generation."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer3_answers import AnswerGenerationLayer

    layer = AnswerGenerationLayer(config, checkpoint)
    result = layer.run()

    print(f"\nLayer 3 Complete: {result.status}")
    print(f"  Answers generated: {result.configs_completed}")
    print(f"  Duration: {result.duration_seconds:.0f}s")


def cmd_layer4(args):
    """Run Layer 4: LLM-as-judge."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer4_judge import JudgeLayer

    layer = JudgeLayer(config, checkpoint)
    result = layer.run()

    print(f"\nLayer 4 Complete: {result.status}")
    print(f"  Queries judged: {result.configs_completed}")
    print(f"  Duration: {result.duration_seconds:.0f}s")


def cmd_layer5(args):
    """Run Layer 5: Analysis + visualization."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer5_analysis import AnalysisLayer

    layer = AnalysisLayer(config, checkpoint)
    result = layer.run()

    print(f"\nLayer 5 Complete: {result.status}")
    print(f"  Duration: {result.duration_seconds:.0f}s")


def cmd_layer6(args):
    """Run Layer 6: Failure categorization."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer6_failures import FailureLayer

    layer = FailureLayer(config, checkpoint)
    result = layer.run()

    print(f"\nLayer 6 Complete: {result.status}")
    print(f"  Duration: {result.duration_seconds:.0f}s")


def cmd_embed(args):
    """Run embedding model shootout."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer_embed import EmbeddingShootoutLayer

    layer = EmbeddingShootoutLayer(config, checkpoint)
    result = layer.run()

    print(f"\n{'=' * 60}")
    print(f"Embedding Shootout Complete: {result.status}")
    print(f"  Models tested: {result.configs_completed}/{result.configs_total}")
    print(f"  Skipped (checkpoint): {result.configs_skipped}")
    if result.best_config_id:
        print(f"  Best: {result.best_config_id} (composite={result.best_score:.4f})")
    print(f"  Duration: {result.duration_seconds:.0f}s")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for e in result.errors:
            print(f"    - {e}")
    print(f"{'=' * 60}")


def cmd_rerank(args):
    """Run reranker model shootout."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer_rerank import RerankerShootoutLayer

    layer = RerankerShootoutLayer(config, checkpoint)
    result = layer.run()

    print(f"\n{'=' * 60}")
    print(f"Reranker Shootout Complete: {result.status}")
    print(f"  Models tested: {result.configs_completed}/{result.configs_total}")
    print(f"  Skipped (checkpoint): {result.configs_skipped}")
    if result.best_config_id:
        print(f"  Best: {result.best_config_id} (composite={result.best_score:.4f})")
    print(f"  Duration: {result.duration_seconds:.0f}s")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for e in result.errors:
            print(f"    - {e}")
    print(f"{'=' * 60}")


def cmd_llm(args):
    """Run chat LLM model comparison."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer_llm import LLMComparisonLayer

    layer = LLMComparisonLayer(config, checkpoint)
    result = layer.run()

    print(f"\n{'=' * 60}")
    print(f"Chat LLM Comparison Complete: {result.status}")
    print(f"  Models tested: {result.configs_completed}/{result.configs_total}")
    print(f"  Skipped (checkpoint): {result.configs_skipped}")
    if result.best_config_id:
        print(f"  Best: {result.best_config_id} (avg {result.best_score:.2f}s/query)")
    print(f"  Duration: {result.duration_seconds:.0f}s")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for e in result.errors:
            print(f"    - {e}")
    print(f"{'=' * 60}")


def cmd_cross(args):
    """Run cross-model sweep."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer_cross import CrossModelSweepLayer

    layer = CrossModelSweepLayer(config, checkpoint)
    result = layer.run()

    print(f"\n{'=' * 60}")
    print(f"Cross-Model Sweep Complete: {result.status}")
    print(f"  Combos tested: {result.configs_completed}/{result.configs_total}")
    print(f"  Skipped (checkpoint): {result.configs_skipped}")
    if result.best_config_id:
        print(f"  Best: {result.best_config_id} (composite={result.best_score:.4f})")
    print(f"  Duration: {result.duration_seconds:.0f}s")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for e in result.errors:
            print(f"    - {e}")
    print(f"{'=' * 60}")


def cmd_live(args):
    """Run live pipeline test via MCP API."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer_live import LivePipelineLayer

    layer = LivePipelineLayer(config, checkpoint)
    result = layer.run()

    print(f"\n{'=' * 60}")
    print(f"Live Pipeline Test: {result.status}")
    if result.summary:
        s = result.summary
        print(f"  Queries tested: {s.get('benchmark_queries_tested', 0)}")
        print(f"  Avg latency: {s.get('avg_latency_ms', 0):.0f}ms")
        print(f"  Avg confidence: {s.get('avg_confidence', 0):.3f}")
        print(f"  Keyword hit rate: {s.get('keyword_hit_rate', 0):.3f}")
        print(f"  Adversarial pass rate: {s.get('adversarial_pass_rate', 0):.1%}")
    print(f"  Duration: {result.duration_seconds:.0f}s")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for e in result.errors:
            print(f"    - {e}")
    print(f"{'=' * 60}")


def cmd_ceiling(args):
    """Run frontier vs local LLM ceiling comparison."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    from benchmark.layers.layer_ceiling import CeilingComparisonLayer

    layer = CeilingComparisonLayer(config, checkpoint)
    result = layer.run()

    print(f"\n{'=' * 60}")
    print(f"Frontier vs Local LLM Ceiling: {result.status}")
    if result.summary:
        s = result.summary
        print(f"  Ceiling avg score: {s.get('ceiling_avg_composite', 0):.3f}")
        print(f"  Local avg score: {s.get('local_avg_composite', 0):.3f}")
        print(f"  Model quality delta: {s.get('model_quality_delta', 0):.3f}")
        print(f"  Queries ceiling wins: {s.get('ceiling_wins', 0)}")
    print(f"  Duration: {result.duration_seconds:.0f}s")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for e in result.errors:
            print(f"    - {e}")
    print(f"{'=' * 60}")


def cmd_full(args):
    """Run all layers sequentially."""
    config = _build_config(args)
    checkpoint = _get_checkpoint(config)

    layers_to_run = [
        ("layer0", "benchmark.layers.layer0_chunking", "ChunkingSweepLayer"),
        ("layer1", "benchmark.layers.layer1_retrieval", "RetrievalSweepLayer"),
        ("layer2", "benchmark.layers.layer2_params", "ParamSweepLayer"),
        ("layer3", "benchmark.layers.layer3_answers", "AnswerGenerationLayer"),
        ("layer4", "benchmark.layers.layer4_judge", "JudgeLayer"),
        ("layer5", "benchmark.layers.layer5_analysis", "AnalysisLayer"),
        ("layer6", "benchmark.layers.layer6_failures", "FailureLayer"),
        ("live", "benchmark.layers.layer_live", "LivePipelineLayer"),
        ("ceiling", "benchmark.layers.layer_ceiling", "CeilingComparisonLayer"),
    ]

    # Check which layers to skip
    start_from = getattr(args, "start_from", None)
    started = start_from is None

    for name, module_path, class_name in layers_to_run:
        if not started:
            if name == start_from:
                started = True
            else:
                continue

        print(f"\n{'=' * 60}")
        print(f"Running {name}...")
        print(f"{'=' * 60}")

        try:
            import importlib

            mod = importlib.import_module(module_path)
            layer_cls = getattr(mod, class_name)
            layer = layer_cls(config, checkpoint)
            result = layer.run()

            print(f"{name}: {result.status} ({result.duration_seconds:.0f}s)")

            if result.status == "failed":
                print(f"  Errors: {result.errors}")
                if not getattr(args, "continue_on_failure", False):
                    print("Stopping. Use --continue-on-failure to proceed.")
                    break
        except Exception as e:
            print(f"{name} FAILED: {e}")
            if not getattr(args, "continue_on_failure", False):
                break


def cmd_status(args):
    """Show benchmark run status."""
    from benchmark.config import BenchmarkConfig

    config = _build_config(args)
    output_dir = Path(config.output_dir)

    if not output_dir.exists():
        print(f"Run not found: {config.run_id}")
        return

    print(f"\nBenchmark Run: {config.run_id}")
    print(f"Output: {output_dir}")

    # Check each layer
    for layer_name in [
        "layer0_chunking",
        "layer1_retrieval",
        "layer2_params",
        "layer_embed",
        "layer_rerank",
        "layer_cross",
        "layer_llm",
        "layer3_answers",
        "layer4_judge",
        "layer5_analysis",
        "layer6_failures",
        "layer_live",
        "layer_ceiling",
    ]:
        result_path = output_dir / f"{layer_name}_result.json"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            status = data.get("status", "unknown")
            duration = data.get("duration_seconds", 0)
            completed = data.get("configs_completed", 0)
            total = data.get("configs_total", 0)
            best = data.get("best_config_id", "")
            score = data.get("best_score", 0)
            print(f"  {layer_name}: {status} ({completed}/{total}, {duration:.0f}s)")
            if best:
                print(f"    Best: {best} ({score:.4f})")
        else:
            print(f"  {layer_name}: not started")

    # Check checkpoints
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        from benchmark.checkpoint import CheckpointManager

        cp = CheckpointManager(checkpoint_dir)
        cp_status = cp.get_status()
        if cp_status:
            print(f"\nCheckpoints:")
            for layer, info in cp_status.items():
                print(f"  {layer}: {info['completed']} completed")


def main():
    parser = argparse.ArgumentParser(
        description="ARCA Benchmark Harness v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Layer to run")

    # Common args
    def add_common(sp):
        sp.add_argument("--run-id", help="Run ID (default: timestamp)")
        sp.add_argument("--topic", default="benchmark", help="Qdrant topic prefix")

    # Layer 0
    p0 = subparsers.add_parser("layer0", help="Chunking configuration sweep")
    add_common(p0)
    p0.add_argument("--corpus", help="Path to .docx corpus directory")
    p0.add_argument(
        "--max-configs", type=int, default=0, help="Max configs to test (0=all)"
    )
    p0.set_defaults(func=cmd_layer0)

    # Layer 1
    p1 = subparsers.add_parser("layer1", help="Retrieval toggle sweep")
    add_common(p1)
    p1.set_defaults(func=cmd_layer1)

    # Layer 2
    p2 = subparsers.add_parser("layer2", help="Continuous parameter sweep")
    add_common(p2)
    p2.set_defaults(func=cmd_layer2)

    # Layer 3
    p3 = subparsers.add_parser("layer3", help="Answer generation")
    add_common(p3)
    p3.set_defaults(func=cmd_layer3)

    # Layer 4
    p4 = subparsers.add_parser("layer4", help="LLM-as-judge scoring")
    add_common(p4)
    p4.set_defaults(func=cmd_layer4)

    # Layer 5
    p5 = subparsers.add_parser("layer5", help="Analysis + visualization")
    add_common(p5)
    p5.set_defaults(func=cmd_layer5)

    # Layer 6
    p6 = subparsers.add_parser("layer6", help="Failure categorization")
    add_common(p6)
    p6.set_defaults(func=cmd_layer6)

    # Embedding shootout
    pe = subparsers.add_parser("embed", help="Embedding model shootout")
    add_common(pe)
    pe.set_defaults(func=cmd_embed)

    # Reranker shootout
    pr = subparsers.add_parser("rerank", help="Reranker model shootout")
    add_common(pr)
    pr.set_defaults(func=cmd_rerank)

    # Cross-model sweep
    px = subparsers.add_parser("cross", help="Cross-model sweep (chunk x embed x rerank)")
    add_common(px)
    px.set_defaults(func=cmd_cross)

    # Chat LLM comparison
    pl = subparsers.add_parser("llm", help="Chat LLM model comparison")
    add_common(pl)
    pl.set_defaults(func=cmd_llm)

    # Live pipeline test
    pol = subparsers.add_parser("live", help="Live pipeline test via MCP API")
    add_common(pol)
    pol.set_defaults(func=cmd_live)

    # Frontier vs local LLM ceiling
    poc = subparsers.add_parser("ceiling", help="Frontier vs local LLM ceiling comparison")
    add_common(poc)
    poc.set_defaults(func=cmd_ceiling)

    # Full run
    pf = subparsers.add_parser("full", help="Run all layers sequentially")
    add_common(pf)
    pf.add_argument("--corpus", help="Path to .docx corpus directory")
    pf.add_argument("--max-configs", type=int, default=0, help="Max L0 configs")
    pf.add_argument("--start-from", help="Start from this layer (skip earlier)")
    pf.add_argument("--continue-on-failure", action="store_true")
    pf.set_defaults(func=cmd_full)

    # Status
    ps = subparsers.add_parser("status", help="Show run status")
    add_common(ps)
    ps.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
