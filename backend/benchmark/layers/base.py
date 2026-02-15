"""
Base Layer — abstract base class for all benchmark layers.
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LayerResult:
    """Result from running a benchmark layer."""

    layer: str
    run_id: str
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    configs_total: int = 0
    configs_completed: int = 0
    configs_skipped: int = 0  # From checkpoint
    best_config_id: Optional[str] = None
    best_score: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": round(self.duration_seconds, 1),
            "configs_total": self.configs_total,
            "configs_completed": self.configs_completed,
            "configs_skipped": self.configs_skipped,
            "best_config_id": self.best_config_id,
            "best_score": round(self.best_score, 4),
            "summary": self.summary,
            "errors": self.errors,
        }

    def save(self, output_dir: Path):
        """Save layer result to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.layer}_result.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        logger.info(f"Saved {self.layer} result to {path}")


class BaseLayer(ABC):
    """Abstract base class for benchmark layers."""

    LAYER_NAME: str = "base"

    def __init__(self, config, checkpoint_mgr):
        """
        Args:
            config: BenchmarkConfig instance
            checkpoint_mgr: CheckpointManager instance
        """
        self.config = config
        self.checkpoint = checkpoint_mgr
        self.output_dir = Path(config.output_dir) / self.LAYER_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> LayerResult:
        """Execute the layer with timing and error handling."""
        from datetime import datetime

        result = LayerResult(
            layer=self.LAYER_NAME,
            run_id=self.config.run_id,
            status="running",
            started_at=datetime.now().isoformat(),
        )

        start = time.time()
        try:
            result = self.execute(result)
            result.status = "completed"
        except Exception as e:
            result.status = "failed"
            result.errors.append(str(e))
            logger.error(f"Layer {self.LAYER_NAME} failed: {e}", exc_info=True)

        result.duration_seconds = time.time() - start
        result.completed_at = datetime.now().isoformat()
        result.save(Path(self.config.output_dir))

        return result

    @abstractmethod
    def execute(self, result: LayerResult) -> LayerResult:
        """Implement the layer logic. Must return the updated LayerResult."""
        ...

    def load_previous_result(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Load result from a previous layer."""
        path = Path(self.config.output_dir) / f"{layer_name}_result.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def load_optimal_config(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Load the optimal config chosen by a previous layer."""
        path = Path(self.config.output_dir) / layer_name / "optimal_config.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None
