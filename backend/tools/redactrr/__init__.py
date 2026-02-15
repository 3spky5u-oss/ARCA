"""
Redactrr - Intelligent Document Redaction

Removes PII from Word and PDF documents for safe external sharing.

Usage:
    from tools.redactrr import redact_document, preview_redaction

    # Preview what would be redacted
    preview = preview_redaction("/path/to/report.docx")
    for entity in preview['entities']:
        print(f"{entity['type']}: {entity['text']}")

    # Redact document
    result = redact_document("/path/to/report.docx")
    print(f"Output: {result.redacted_path}")

CLI:
    python -m tools.redactrr preview report.docx
    python -m tools.redactrr redact report.docx
    python -m tools.redactrr batch ./reports/ --output-dir ./clean/
"""

from .redactor import (
    Redactrr,
    RedactionResult,
    PIIEntity,
    PIIType,
    PIIDetector,
    redact_document,
    preview_redaction,
)

__all__ = [
    "Redactrr",
    "RedactionResult",
    "PIIEntity",
    "PIIType",
    "PIIDetector",
    "redact_document",
    "preview_redaction",
]
