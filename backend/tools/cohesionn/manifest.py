"""
Manifest tracking for ingested knowledge files.

Tracks which files have been ingested into Qdrant by content hash,
enabling detection of new, changed, and deleted files on startup.
"""

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class IngestedFile:
    """Record of an ingested file"""

    file_path: str
    file_hash: str
    topic: str
    ingested_at: str
    chunks_created: int
    file_size: int


class IngestManifest:
    """
    Tracks ingested files by content hash.

    Enables detection of:
    - New files (not in manifest)
    - Changed files (hash mismatch)
    - Deleted files (in manifest but not on disk)
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.files: Dict[str, IngestedFile] = {}
        self._load()

    def _load(self):
        """Load manifest from disk"""
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self.files = {k: IngestedFile(**v) for k, v in data.get("files", {}).items()}
                logger.debug(f"Loaded manifest with {len(self.files)} entries")
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
                self.files = {}

    def save(self):
        """Save manifest to disk"""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "files": {k: asdict(v) for k, v in self.files.items()},
        }
        self.manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug(f"Saved manifest with {len(self.files)} entries")

    @staticmethod
    def compute_hash(file_path: Path) -> str:
        """Compute MD5 hash of file content"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def is_ingested(self, file_path: Path) -> bool:
        """Check if file has already been ingested (by content hash)"""
        if not file_path.exists():
            return False
        file_hash = self.compute_hash(file_path)
        return file_hash in self.files

    def needs_reingestion(self, file_path: Path) -> bool:
        """
        Check if file needs re-ingestion.

        Returns True if:
        - File is not in manifest (new file)
        - File hash has changed (content modified)
        """
        if not file_path.exists():
            return False
        file_hash = self.compute_hash(file_path)
        return file_hash not in self.files

    def record(self, file_path: Path, topic: str, chunks: int):
        """Record a successfully ingested file"""
        file_hash = self.compute_hash(file_path)
        self.files[file_hash] = IngestedFile(
            file_path=str(file_path),
            file_hash=file_hash,
            topic=topic,
            ingested_at=datetime.now().isoformat(),
            chunks_created=chunks,
            file_size=file_path.stat().st_size,
        )

    def get_stale_hashes(self, knowledge_dir: Path) -> List[str]:
        """
        Find manifest entries for files that no longer exist.

        Args:
            knowledge_dir: Base knowledge directory to check against

        Returns:
            List of file hashes that are stale (file deleted)
        """
        stale = []
        for file_hash, entry in self.files.items():
            if not Path(entry.file_path).exists():
                stale.append(file_hash)
        return stale

    def remove_stale(self, stale_hashes: List[str]):
        """Remove stale entries from manifest"""
        for h in stale_hashes:
            if h in self.files:
                logger.debug(f"Removing stale entry: {self.files[h].file_path}")
                del self.files[h]

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about ingested files"""
        stats = {"total_files": len(self.files), "total_chunks": 0, "by_topic": {}}
        for entry in self.files.values():
            stats["total_chunks"] += entry.chunks_created
            topic = entry.topic
            if topic not in stats["by_topic"]:
                stats["by_topic"][topic] = {"files": 0, "chunks": 0}
            stats["by_topic"][topic]["files"] += 1
            stats["by_topic"][topic]["chunks"] += entry.chunks_created
        return stats
