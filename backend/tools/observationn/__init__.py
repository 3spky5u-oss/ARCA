"""
Observationn - Qwen3-VL Vision-Based Document Extraction

Uses Qwen3-VL (or compatible vision LLM) for:
- High-quality OCR of scanned documents
- Table structure recognition
- Diagram and figure understanding
- Equation extraction
"""

from .extractor import ObservationnExtractor

__all__ = ["ObservationnExtractor"]
