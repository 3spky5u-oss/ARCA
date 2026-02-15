"""
Checkpoint Manager — crash recovery for long-running benchmark layers.

Saves per-config results as JSON. On restart, completed configs are skipped.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Per-layer checkpoint persistence."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._completed: Dict[str, Set[str]] = {}  # layer -> set of config_ids
        self._load_existing()

    def _checkpoint_path(self, layer: str) -> Path:
        return self.checkpoint_dir / f"{layer}_checkpoint.json"

    def _load_existing(self):
        """Load completed config IDs from all checkpoint files."""
        for f in self.checkpoint_dir.glob("*_checkpoint.json"):
            layer = f.stem.replace("_checkpoint", "")
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self._completed[layer] = set(data.get("completed_ids", []))
                logger.info(f"Checkpoint loaded: {layer} has {len(self._completed[layer])} completed configs")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {f}: {e}")

    def is_completed(self, layer: str, config_id: str) -> bool:
        return config_id in self._completed.get(layer, set())

    def get_completed(self, layer: str) -> Set[str]:
        return self._completed.get(layer, set()).copy()

    def mark_completed(self, layer: str, config_id: str, result: Dict[str, Any]):
        """Mark a config as completed and save its result."""
        if layer not in self._completed:
            self._completed[layer] = set()
        self._completed[layer].add(config_id)

        # Save individual result
        result_dir = self.checkpoint_dir / layer
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{config_id}.json"
        result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

        # Update checkpoint index
        self._save_checkpoint(layer)

    def _save_checkpoint(self, layer: str):
        path = self._checkpoint_path(layer)
        data = {
            "completed_ids": sorted(self._completed.get(layer, set())),
            "count": len(self._completed.get(layer, set())),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_result(self, layer: str, config_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved result for a specific config."""
        result_path = self.checkpoint_dir / layer / f"{config_id}.json"
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        return None

    def get_all_results(self, layer: str) -> List[Dict[str, Any]]:
        """Load all saved results for a layer."""
        result_dir = self.checkpoint_dir / layer
        if not result_dir.exists():
            return []
        results = []
        for f in sorted(result_dir.glob("*.json")):
            try:
                results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"Failed to load result {f}: {e}")
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get checkpoint status across all layers."""
        return {
            layer: {
                "completed": len(ids),
                "ids": sorted(ids)[:10],  # First 10 for preview
            }
            for layer, ids in self._completed.items()
        }
