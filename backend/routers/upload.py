"""
Upload Router - 100% RAM-based, accepts ALL file types

No disk writes. Everything stored in memory until session ends or server restarts.
"""

import logging
import io
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Security: Maximum file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


# =============================================================================
# File Type Detection
# =============================================================================


class FileType(Enum):
    EXCEL = "excel"
    CSV = "csv"
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    OTHER = "other"


# Security: Only allow document MIME types, not code files
MIME_MAP = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileType.EXCEL,
    "application/vnd.ms-excel": FileType.EXCEL,
    "application/vnd.ms-excel.sheet.macroEnabled.12": FileType.EXCEL,
    "text/csv": FileType.CSV,
    "application/csv": FileType.CSV,
    "application/pdf": FileType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.WORD,
    "application/msword": FileType.WORD,
    "text/plain": FileType.TEXT,
    "text/markdown": FileType.TEXT,
    "image/png": FileType.IMAGE,
    "image/jpeg": FileType.IMAGE,
    "image/gif": FileType.IMAGE,
    # Code MIME types removed for security - text/x-python, application/javascript, application/json
}

# Security: Only allow document types, not code files
EXT_MAP = {
    ".xlsx": FileType.EXCEL,
    ".xls": FileType.EXCEL,
    ".xlsm": FileType.EXCEL,
    ".csv": FileType.CSV,
    ".tsv": FileType.CSV,
    ".pdf": FileType.PDF,
    ".docx": FileType.WORD,
    ".doc": FileType.WORD,
    ".txt": FileType.TEXT,
    ".md": FileType.TEXT,
    ".log": FileType.TEXT,
    ".png": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".gif": FileType.IMAGE,
    # Code files removed for security - .py, .js, .html, .css, .json, .xml, .yaml, .yml, .sql, .sh, .bat, .ps1
}


def detect_type(filename: str, mime: str = None) -> FileType:
    if mime and mime in MIME_MAP:
        return MIME_MAP[mime]
    ext = Path(filename).suffix.lower()
    return EXT_MAP.get(ext, FileType.OTHER)


# =============================================================================
# RAM Storage
# =============================================================================


@dataclass
class StoredFile:
    """File stored entirely in RAM"""

    file_id: str
    filename: str
    data: bytes  # The actual file content
    file_type: FileType
    mime_type: str
    size: int
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For lab data files
    lab_data: Any = None
    samples: int = 0
    parameters: int = 0
    analysis: Any = None

    # For RAG-indexed files
    rag_chunks: int = 0

    def get_bytesio(self) -> io.BytesIO:
        """Get file as BytesIO for processing"""
        return io.BytesIO(self.data)


# Global RAM storage - everything lives here
files_db: Dict[str, StoredFile] = {}

# Session RAG storage (in-memory vector store)
_session_rag = None


def get_session_rag() -> Any:
    """Get or create in-memory RAG for session. Currently disabled (needs Qdrant migration)."""
    return None


# =============================================================================
# Response Models
# =============================================================================


class UploadResponse(BaseModel):
    success: bool
    file_id: str
    filename: str
    file_type: str
    size: int
    # Lab data (if applicable)
    samples: int = 0
    parameters: int = 0
    # RAG (if applicable)
    rag_chunks: int = 0


class FileListItem(BaseModel):
    file_id: str
    filename: str
    file_type: str
    size: int
    created_at: str


# =============================================================================
# Helper Functions (for other modules)
# =============================================================================


def get_file_data(file_id: str) -> Optional[StoredFile]:
    """Get file data by ID or filename"""
    # Direct ID lookup
    if file_id in files_db:
        return files_db[file_id]

    # Fallback: search by filename (LLM often passes filename instead of ID)
    for stored in files_db.values():
        if stored.filename == file_id:
            return stored

    return None


def get_file_bytes(file_id: str) -> Optional[bytes]:
    """Get raw bytes"""
    f = files_db.get(file_id)
    return f.data if f else None


def get_file_bytesio(file_id: str) -> Optional[io.BytesIO]:
    """Get as BytesIO"""
    f = files_db.get(file_id)
    return f.get_bytesio() if f else None


def update_file_analysis(file_id: str, analysis_result: Dict[str, Any]) -> None:
    """Update file with analysis results"""
    if file_id in files_db:
        files_db[file_id].analysis = analysis_result


def store_generated_file(filename: str, data: bytes, mime_type: str = None) -> str:
    """Store a generated file (reports, redacted docs)"""
    file_id = hashlib.md5(f"gen_{filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    files_db[file_id] = StoredFile(
        file_id=file_id,
        filename=filename,
        data=data,
        file_type=detect_type(filename, mime_type),
        mime_type=mime_type or "application/octet-stream",
        size=len(data),
        created_at=datetime.now().isoformat(),
        metadata={"generated": True},
    )

    return file_id


def clear_files() -> None:
    """Clear all uploaded files from memory"""
    global files_db, _session_rag
    files_db.clear()
    _session_rag = None
    logger.info("Cleared all session files from RAM")


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file - stored in RAM"""
    return await _process_upload([file])


@router.post("/upload/multi", response_model=UploadResponse)
async def upload_multiple(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    return await _process_upload(files)


@router.get("/files")
async def list_files():
    """List all uploaded files"""
    return {
        "success": True,
        "files": [
            FileListItem(
                file_id=f.file_id,
                filename=f.filename,
                file_type=f.file_type.value,
                size=f.size,
                created_at=f.created_at,
            )
            for f in files_db.values()
        ],
        "total_mb": round(sum(f.size for f in files_db.values()) / 1024 / 1024, 2),
    }


@router.get("/files/{file_id}")
async def get_file(file_id: str):
    """Get file info"""
    f = files_db.get(file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "file_id": f.file_id,
        "filename": f.filename,
        "file_type": f.file_type.value,
        "size": f.size,
        "created_at": f.created_at,
        "samples": f.samples,
        "parameters": f.parameters,
        "rag_chunks": f.rag_chunks,
    }


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file"""
    if file_id in files_db:
        del files_db[file_id]
    return {"success": True}


@router.delete("/files")
async def delete_all_files():
    """Clear all files"""
    clear_files()
    return {"success": True, "message": "All files cleared from RAM"}


# =============================================================================
# Upload Processing
# =============================================================================


async def _process_upload(files: List[UploadFile]) -> UploadResponse:
    """Process uploaded files - store in RAM, parse if applicable"""

    # Read all files once and validate (no double-read)
    file_data_list: List[tuple] = []  # (file, data, ext)
    for file in files:
        # Security: Check file size before reading into memory
        if file.size and file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: {file.filename} exceeds 50MB limit")

        data = await file.read()

        # Secondary check after read (file.size may not always be set)
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: {file.filename} exceeds 50MB limit")

        # Security: Check file extension
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in EXT_MAP:
            raise HTTPException(status_code=415, detail=f"File type not allowed: {ext}")

        file_data_list.append((file, data, ext))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_id = hashlib.md5(f"{timestamp}_{files[0].filename}".encode()).hexdigest()[:12]

    total_size = 0
    total_samples = 0
    total_params = 0
    total_rag_chunks = 0
    filenames = []
    primary_type = None
    combined_data = b""

    for file, data, ext in file_data_list:
        # Data already read - no second read needed
        total_size += len(data)
        filenames.append(file.filename)

        file_type = detect_type(file.filename, file.content_type)
        if primary_type is None:
            primary_type = file_type
            combined_data = data

        logger.info(f"Uploaded to RAM: {file.filename} ({len(data)} bytes, {file_type.value})")

        # Store in RAM
        stored = StoredFile(
            file_id=file_id if len(files) == 1 else f"{file_id}_{len(filenames)}",
            filename=file.filename,
            data=data,
            file_type=file_type,
            mime_type=file.content_type or "application/octet-stream",
            size=len(data),
            created_at=datetime.now().isoformat(),
        )

        # Process based on type
        if file_type in [FileType.EXCEL, FileType.CSV]:
            samples, params = _try_parse_lab_data(stored)
            stored.samples = samples
            stored.parameters = params
            total_samples += samples
            total_params += params

        elif file_type in [FileType.PDF, FileType.WORD, FileType.TEXT]:
            chunks = _try_add_to_rag(stored)
            stored.rag_chunks = chunks
            total_rag_chunks += chunks

        # Store in global registry
        files_db[stored.file_id] = stored

    # For multi-file uploads, also store combined reference
    if len(files) > 1:
        files_db[file_id] = StoredFile(
            file_id=file_id,
            filename=", ".join(filenames),
            data=combined_data,
            file_type=primary_type,
            mime_type="application/octet-stream",
            size=total_size,
            created_at=datetime.now().isoformat(),
            samples=total_samples,
            parameters=total_params,
            rag_chunks=total_rag_chunks,
        )

    return UploadResponse(
        success=True,
        file_id=file_id,
        filename=", ".join(filenames),
        file_type=primary_type.value if primary_type else "other",
        size=total_size,
        samples=total_samples,
        parameters=total_params,
        rag_chunks=total_rag_chunks,
    )


def _try_parse_lab_data(stored: StoredFile) -> tuple:
    """Try to parse as lab data, return (samples, params)"""
    try:
        from tools.exceedee.parser import parse_bv_files
        import tempfile
        import os

        # Exceedee needs file paths - use temp file (gets auto-deleted)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(stored.filename).suffix) as tmp:
            tmp.write(stored.data)
            tmp_path = tmp.name

        try:
            lab_data = parse_bv_files([tmp_path])
            stored.lab_data = lab_data

            samples = len(lab_data) if lab_data else 0
            params = len(lab_data.get_all_parameters()) if lab_data else 0

            logger.info(f"Parsed lab data: {samples} samples, {params} params")
            return samples, params
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except ImportError:
        # Exceedee not installed
        return 0, 0
    except Exception as e:
        # Not valid lab data - that's OK
        logger.debug(f"Not lab data ({stored.filename}): {e}")
        return 0, 0


def _try_add_to_rag(stored: StoredFile) -> int:
    """Try to add to session RAG, return chunk count"""
    try:
        # Extract text
        text = _extract_text(stored)
        if not text or len(text.strip()) < 50:
            return 0

        # Get embedder
        from tools.cohesionn.embeddings import get_embedder
        from tools.cohesionn.chunker import SemanticChunker

        embedder = get_embedder()
        chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)

        # Chunk
        chunks = chunker.chunk_text(
            text,
            {
                "filename": stored.filename,
                "file_id": stored.file_id,
            },
        )

        if not chunks:
            return 0

        # Get session RAG collection
        rag = get_session_rag()
        if rag is None:
            return 0

        # Embed and store
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_documents(texts)

        rag.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.metadata for c in chunks],
        )

        logger.info(f"Added {len(chunks)} chunks to session RAG: {stored.filename}")
        return len(chunks)

    except ImportError as e:
        logger.debug(f"RAG not available: {e}")
        return 0
    except Exception as e:
        logger.warning(f"RAG indexing failed: {e}")
        return 0


def _extract_text(stored: StoredFile) -> str:
    """Extract text from stored file"""
    try:
        if stored.file_type == FileType.PDF:
            import fitz

            doc = fitz.open(stream=stored.data, filetype="pdf")
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text

        elif stored.file_type == FileType.WORD:
            from docx import Document

            doc = Document(stored.get_bytesio())
            return "\n\n".join(para.text for para in doc.paragraphs)

        elif stored.file_type == FileType.TEXT:
            return stored.data.decode("utf-8", errors="ignore")

    except Exception as e:
        logger.warning(f"Text extraction failed: {e}")

    return ""


# =============================================================================
# Session RAG Search (for chat.py to use)
# =============================================================================


def _normalize_score(raw_score: float) -> float:
    """Normalize raw cosine similarity to intuitive 0-1 range.

    Raw BGE-M3 cosine similarity typically ranges 0.1-0.5 for good matches.
    This function maps that to a more intuitive 30%-98% display range.
    """
    if raw_score >= 0.40:
        return min(0.98, 0.90 + (raw_score - 0.40) * 0.2)
    elif raw_score >= 0.25:
        return 0.75 + (raw_score - 0.25) * 1.0
    elif raw_score >= 0.15:
        return 0.55 + (raw_score - 0.15) * 2.0
    elif raw_score >= 0.05:
        return 0.30 + (raw_score - 0.05) * 2.5
    else:
        return raw_score * 6.0


def _clean_filename_for_display(filename: str) -> str:
    """Convert filename to readable title for citations.

    Transforms: 'Technical_Report_Final_Draft.pdf'
    Into: 'Technical Report Final Draft'
    """
    import re

    name = Path(filename).stem  # Remove extension
    name = name.replace("_", " ")  # Replace underscores with spaces
    name = name.replace("-", " ")  # Replace hyphens with spaces
    name = re.sub(r"\s+", " ", name).strip()  # Collapse multiple spaces

    # Truncate if too long
    if len(name) > 60:
        name = name[:57] + "..."

    return name if name else "Document"


def search_session_docs(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search session documents"""
    rag = get_session_rag()

    if rag is None or rag.count() == 0:
        return {
            "success": False,
            "error": "No documents uploaded to search",
            "results": [],
        }

    try:
        from tools.cohesionn.embeddings import get_embedder

        embedder = get_embedder()

        query_emb = embedder.embed_query(query)

        results = rag.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, rag.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return {"success": True, "results": [], "context": ""}

        # Format results
        formatted = []
        context_parts = []

        for i in range(len(results["ids"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            raw_score = 1 - results["distances"][0][i]
            normalized_score = _normalize_score(raw_score)

            # Clean filename for display
            raw_filename = meta.get("filename", "Unknown")
            clean_filename = _clean_filename_for_display(raw_filename)

            formatted.append(
                {
                    "content": doc[:500] + "..." if len(doc) > 500 else doc,
                    "filename": clean_filename,
                    "raw_filename": raw_filename,  # Keep original for file lookups
                    "score": round(normalized_score, 3),
                    "raw_score": round(raw_score, 3),  # Keep raw for debugging
                }
            )

            source = f"[{clean_filename}]"
            context_parts.append(f"--- {source} ---\n{doc}")

        return {
            "success": True,
            "results": formatted,
            "context": "\n\n".join(context_parts),
            "num_results": len(formatted),
        }

    except Exception as e:
        logger.error(f"Session search failed: {e}")
        return {"success": False, "error": str(e), "results": []}
