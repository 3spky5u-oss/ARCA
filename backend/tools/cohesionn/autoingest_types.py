"""
Shared types and constants for the auto-ingestion pipeline.

Extracted from autoingest.py to keep the main module focused on
orchestration (AutoIngestService) while phase logic lives in
autoingest_phases.py.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]

# RAM pressure threshold: spill to disk if available memory < 25% of total
_RAM_PRESSURE_THRESHOLD = 0.25


@dataclass
class PhasedFileState:
    """Intermediate state for a file during three-phase batch ingest."""

    file_path: Path
    topic: str
    text_pages: List[Dict]  # {page_num, text, content_type}
    vision_page_specs: List[tuple] = field(default_factory=list)  # [(page_num, classification), ...]
    page_classifications: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    full_text: str = ""
    _spilled_path: Optional[Path] = None  # NVMe spillover path if RAM-constrained

    def spill_to_disk(self, temp_dir: Path) -> None:
        """Serialize text_pages to disk to free RAM."""
        if self._spilled_path is not None:
            return  # Already spilled
        spill_file = temp_dir / f"{self.file_path.stem}_{id(self)}.json"
        spill_file.write_text(
            json.dumps(self.text_pages, ensure_ascii=False),
            encoding="utf-8",
        )
        self._spilled_path = spill_file
        self.text_pages = []  # Free RAM
        self.full_text = ""
        logger.debug(f"Spilled {self.file_path.name} to {spill_file}")

    def load_from_disk(self) -> None:
        """Reload text_pages from disk if spilled."""
        if self._spilled_path is None:
            return
        self.text_pages = json.loads(
            self._spilled_path.read_text(encoding="utf-8")
        )
        self.full_text = "\n\n".join(p["text"] for p in self.text_pages if p.get("text"))
        self._spilled_path.unlink(missing_ok=True)
        self._spilled_path = None

    def cleanup(self) -> None:
        """Remove spillover file if it exists."""
        if self._spilled_path and self._spilled_path.exists():
            self._spilled_path.unlink(missing_ok=True)
