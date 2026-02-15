"""
Graph Extractor - Extract chart data as structured JSON

Converts charts and graphs into searchable structured data.

Full-page approach: renders entire page at 200 DPI and sends to vision model.
Region detection is avoided because charts are often vector graphics
that PyMuPDF cannot detect as image regions.

Vision model runs on a dedicated llama-server slot alongside the MoE chat model.
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

from config import runtime_config
from utils.llm import get_llm_client
from services.json_repair import parse_json_response

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class FigureData:
    """A single figure/chart extracted from a page."""

    figure_id: str  # e.g., "Fig. 2.1a"
    chart_type: str  # "line", "scatter", "bar", "area", "classification", "contour"
    title: str

    # Axes
    x_axis: Dict[str, Any]  # {"label", "unit", "scale", "range"}
    y_axis: Dict[str, Any]

    # Data series (multiple curves per chart)
    data_series: List[Dict[str, Any]]  # [{"name", "points": [[x,y], ...], "style"}]

    # Classification zones (labeled regions on chart)
    zones: List[Dict[str, Any]] = field(default_factory=list)  # [{"name", "boundary_points"}]

    # Annotations and equations
    annotations: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)  # LaTeX format

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "chart_type": self.chart_type,
            "title": self.title,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "data_series": self.data_series,
            "zones": self.zones,
            "annotations": self.annotations,
            "equations": self.equations,
        }


@dataclass
class PageChartData:
    """All structured data extracted from a chart page."""

    source_page: int
    figures: List[FigureData] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)  # Inline tables on chart pages
    text_context: str = ""  # Surrounding body text
    confidence: float = 0.8

    def to_searchable_text(self) -> str:
        """
        Generate a text summary for vector search indexing.

        Includes axis labels, series names, zone names, key data ranges
        so that text queries about chart content find chart pages.
        """
        parts = []

        for fig in self.figures:
            parts.append(f"Figure: {fig.title}")
            parts.append(f"Type: {fig.chart_type}")

            x_axis = fig.x_axis or {}
            y_axis = fig.y_axis or {}
            x_label = x_axis.get("label", "")
            x_unit = x_axis.get("unit", "")
            y_label = y_axis.get("label", "")
            y_unit = y_axis.get("unit", "")

            if x_label:
                parts.append(f"X-axis: {x_label}" + (f" ({x_unit})" if x_unit else ""))
            if y_label:
                parts.append(f"Y-axis: {y_label}" + (f" ({y_unit})" if y_unit else ""))

            # Series names and point counts
            for series in fig.data_series:
                name = series.get("name", "unnamed")
                points = series.get("points", [])
                parts.append(f"Series: {name} ({len(points)} points)")

                # Include data range for searchability
                if points:
                    x_vals = [p[0] for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
                    y_vals = [p[1] for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
                    if x_vals and y_vals:
                        parts.append(f"  X range: {min(x_vals):.3g} to {max(x_vals):.3g}")
                        parts.append(f"  Y range: {min(y_vals):.3g} to {max(y_vals):.3g}")

            # Zone names
            for zone in fig.zones:
                parts.append(f"Zone: {zone.get('name', '')}")

            # Equations
            for eq in fig.equations:
                parts.append(f"Equation: {eq}")

            # Annotations
            for ann in fig.annotations:
                parts.append(f"Note: {ann}")

        if self.text_context:
            parts.append(f"Context: {self.text_context}")

        return "\n".join(parts)

    def to_json_chunk(self) -> str:
        """Serialize full structured JSON for storage and recreation."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "chart_data",
            "source_page": self.source_page,
            "figures": [f.to_dict() for f in self.figures],
            "tables": self.tables,
            "text_context": self.text_context,
            "confidence": self.confidence,
        }


# =============================================================================
# Prompts
# =============================================================================

CHART_EXTRACTION_PROMPT_KB = """You are extracting structured data from an engineering chart/graph page.

Output ONLY valid JSON (no other text) in this exact schema:

```json
{
  "figures": [
    {
      "figure_id": "Fig. X.Y",
      "chart_type": "line|scatter|bar|area|classification|contour",
      "title": "Full chart title",
      "x_axis": {
        "label": "Axis label text",
        "unit": "unit (kPa, m, degrees, etc.)",
        "scale": "linear|log",
        "range": [min_value, max_value]
      },
      "y_axis": {
        "label": "Axis label text",
        "unit": "unit",
        "scale": "linear|log",
        "range": [min_value, max_value]
      },
      "data_series": [
        {
          "name": "Series/curve name or label",
          "points": [[x1,y1], [x2,y2], ...],
          "style": "solid|dashed|dotted|markers"
        }
      ],
      "zones": [
        {
          "name": "Zone name or label",
          "boundary_points": [[x1,y1], [x2,y2], ...]
        }
      ],
      "annotations": ["Text annotations on the chart"],
      "equations": ["LaTeX equations shown on chart (e.g., y = ax^2 + bx + c)"]
    }
  ],
  "tables": [
    {
      "title": "Table title",
      "headers": ["Col1", "Col2"],
      "rows": [["val1", "val2"]]
    }
  ],
  "text_context": "Any body text surrounding the chart on this page"
}
```

CRITICAL RULES:
- If the page has multiple subfigures (a, b, c), extract EACH as a separate figure entry
- For EACH data series/curve, extract 8-15 data points minimum to capture the curve shape
- For log-scale axes: space points logarithmically (1, 2, 5, 10, 20, 50, 100...)
- For classification charts with labeled zones: extract zone boundaries as boundary_points
- Include ALL visible curves/series, not just the primary one
- Equations in LaTeX format (use backslash for special symbols)
- Read axis values carefully — check tick marks and gridlines
- Use null for any field you cannot determine
- Output ONLY the JSON, no explanations"""

CHART_EXTRACTION_PROMPT_SESSION = """Extract chart data from this engineering figure as JSON.

Output ONLY valid JSON:
{
  "figures": [
    {
      "figure_id": "Fig. X",
      "chart_type": "line|scatter|bar",
      "title": "Chart title",
      "x_axis": {"label": "X", "unit": "", "scale": "linear", "range": [0, 100]},
      "y_axis": {"label": "Y", "unit": "", "scale": "linear", "range": [0, 100]},
      "data_series": [
        {"name": "Series 1", "points": [[x,y], ...]}
      ]
    }
  ]
}

Extract at least 5 data points per series. Use null for unknown fields."""


# =============================================================================
# Extractor
# =============================================================================


class GraphExtractor:
    """
    Extract chart/graph data as structured JSON.

    Uses vision model to analyze charts and extract:
    - Multi-figure decomposition (subfigures a, b, c)
    - Multiple data series per chart
    - Axis labels, units, and scale (linear/log)
    - Classification zones and labeled regions
    - Equations in LaTeX format
    - 8-15 data points per series minimum

    Full-page approach: renders entire page at 200 DPI.
    Region detection is skipped — vector graphics break PyMuPDF image detection.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dpi: int = 200,
        num_ctx: int = 4096,
    ):
        """
        Args:
            model: Vision model override (default from config based on mode)
            dpi: Resolution for page rendering (200 for chart clarity)
            num_ctx: Context window size (4096 for KB, 2048 for session)
        """
        self.model = model or runtime_config.model_vision_structured
        self.dpi = dpi
        self.num_ctx = num_ctx

    def extract_page_charts(
        self,
        file_path: Path,
        page_num: int,
        prompt: Optional[str] = None,
    ) -> Optional[PageChartData]:
        """
        Extract all chart data from a full page.

        Full-page approach: renders entire page and sends to 32b vision model.
        No region detection — engineering charts are often vector graphics
        that PyMuPDF cannot detect as image regions.

        Args:
            file_path: Path to PDF file
            page_num: Page number (1-indexed)
            prompt: Optional prompt override (default uses KB prompt)

        Returns:
            PageChartData if extraction successful, None otherwise
        """
        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        if page_num < 1 or page_num > len(doc):
            doc.close()
            logger.error(f"Page {page_num} out of range")
            return None

        page = doc[page_num - 1]

        # Render full page at 200 DPI
        pix = page.get_pixmap(dpi=self.dpi)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        doc.close()

        # Select prompt
        extraction_prompt = prompt or CHART_EXTRACTION_PROMPT_KB

        # Send to vision model on dedicated slot
        client = get_llm_client("vision")

        try:
            response = client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": extraction_prompt,
                    "images": [img_b64],
                }],
                options={
                    "num_ctx": self.num_ctx,
                    "temperature": 0.1,
                },
            )

            response_text = response["message"]["content"]
            logger.info(f"Chart extraction response: {len(response_text)} chars for page {page_num}")

            # Parse using shared JSON repair pipeline
            data = parse_json_response(response_text)
            if data is None:
                logger.warning(f"Failed to parse chart JSON for page {page_num}")
                return None

            return self._build_page_chart_data(data, page_num)

        except Exception as e:
            logger.error(f"Chart extraction failed for page {page_num}: {e}")
            return None

    def _build_page_chart_data(self, data: Dict[str, Any], page_num: int) -> Optional[PageChartData]:
        """Build PageChartData from parsed JSON."""
        figures = []

        for fig_data in data.get("figures", []):
            try:
                fig = FigureData(
                    figure_id=fig_data.get("figure_id", f"Fig. {page_num}"),
                    chart_type=fig_data.get("chart_type", "unknown"),
                    title=fig_data.get("title", "Untitled"),
                    x_axis=fig_data.get("x_axis", {}),
                    y_axis=fig_data.get("y_axis", {}),
                    data_series=fig_data.get("data_series", []),
                    zones=fig_data.get("zones", []),
                    annotations=fig_data.get("annotations", []),
                    equations=fig_data.get("equations", []),
                )
                figures.append(fig)
            except Exception as e:
                logger.warning(f"Failed to parse figure on page {page_num}: {e}")

        if not figures:
            logger.warning(f"No figures extracted from page {page_num}")
            return None

        return PageChartData(
            source_page=page_num,
            figures=figures,
            tables=data.get("tables", []),
            text_context=data.get("text_context", ""),
            confidence=0.8,
        )

    # Legacy methods preserved for backward compatibility

    def extract_chart(
        self,
        file_path: Path,
        page_num: int,
        image_rect: Optional[fitz.Rect] = None,
    ) -> Optional["ChartData"]:
        """
        Legacy: Extract chart data from a specific page or region.

        Preserved for backward compatibility. New code should use extract_page_charts().
        """
        result = self.extract_page_charts(file_path, page_num)
        if result is None or not result.figures:
            return None

        # Convert first figure to legacy ChartData format
        fig = result.figures[0]
        # Flatten data_series points into legacy format
        data_points = []
        for series in fig.data_series:
            for pt in series.get("points", []):
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    data_points.append({"x": pt[0], "y": pt[1]})

        return ChartData(
            chart_type=fig.chart_type,
            title=fig.title,
            source_page=page_num,
            x_axis=fig.x_axis,
            y_axis=fig.y_axis,
            data_points=data_points,
            curve_type="interpolated",
            source_text=fig.annotations[0] if fig.annotations else None,
            notes=fig.annotations,
        )

    def extract_charts_from_page(
        self,
        file_path: Path,
        page_num: int,
    ) -> List["ChartData"]:
        """
        Legacy: Extract all charts from a page.

        Preserved for backward compatibility. New code should use extract_page_charts().
        """
        result = self.extract_page_charts(file_path, page_num)
        if result is None:
            return []

        charts = []
        for fig in result.figures:
            data_points = []
            for series in fig.data_series:
                for pt in series.get("points", []):
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        data_points.append({"x": pt[0], "y": pt[1]})

            charts.append(ChartData(
                chart_type=fig.chart_type,
                title=fig.title,
                source_page=page_num,
                x_axis=fig.x_axis,
                y_axis=fig.y_axis,
                data_points=data_points,
                curve_type="interpolated",
                source_text=fig.annotations[0] if fig.annotations else None,
                notes=fig.annotations,
            ))

        return charts


# =============================================================================
# Legacy ChartData (backward compat)
# =============================================================================


@dataclass
class ChartData:
    """Legacy structured data extracted from an engineering chart."""

    chart_type: str
    title: str
    source_page: int
    x_axis: Dict[str, Any]
    y_axis: Dict[str, Any]
    data_points: List[Dict[str, float]]
    curve_type: str
    source_text: Optional[str]
    notes: List[str] = field(default_factory=list)
    confidence: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "chart",
            "chart_type": self.chart_type,
            "title": self.title,
            "source_page": self.source_page,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "data_points": self.data_points,
            "curve_type": self.curve_type,
            "source_text": self.source_text,
            "notes": self.notes,
            "confidence": self.confidence,
        }

    def to_searchable_text(self) -> str:
        """Convert chart data to searchable text representation."""
        parts = [
            f"Chart: {self.title}",
            f"Type: {self.chart_type} chart",
            f"X-axis: {self.x_axis.get('label', 'unknown')} ({self.x_axis.get('unit', '')})",
            f"Y-axis: {self.y_axis.get('label', 'unknown')} ({self.y_axis.get('unit', '')})",
        ]

        if self.source_text:
            parts.append(f"Source: {self.source_text}")

        if self.data_points:
            parts.append(f"Data points: {len(self.data_points)}")
            sample_points = self.data_points[:5]
            for pt in sample_points:
                parts.append(
                    f"  {self.x_axis.get('label', 'x')}={pt.get('x')}: {self.y_axis.get('label', 'y')}={pt.get('y')}"
                )

        if self.notes:
            parts.append("Notes: " + "; ".join(self.notes))

        return "\n".join(parts)

    def to_json_chunk(self) -> str:
        """Convert to JSON string for embedding in chunks."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Table Extractor (unchanged)
# =============================================================================


@dataclass
class TableData:
    """Structured data extracted from a table."""

    title: str
    source_page: int
    headers: List[str]
    rows: List[List[str]]
    column_types: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "table",
            "title": self.title,
            "source_page": self.source_page,
            "headers": self.headers,
            "rows": self.rows,
            "column_types": self.column_types,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        """Convert to markdown table format."""
        if not self.headers:
            return ""

        lines = []
        if self.title:
            lines.append(f"**{self.title}**\n")

        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")

        for row in self.rows:
            padded = row + [""] * (len(self.headers) - len(row))
            lines.append("| " + " | ".join(str(cell) for cell in padded) + " |")

        return "\n".join(lines)


class TableExtractor:
    """Extract table data as structured format."""

    TABLE_EXTRACTION_PROMPT = """Analyze this table and extract its data.

Output ONLY valid JSON in this exact format (no other text):
{{
    "title": "Table title if visible",
    "headers": ["Column 1", "Column 2", ...],
    "rows": [
        ["cell1", "cell2", ...],
        ["cell1", "cell2", ...],
        ...
    ],
    "column_types": ["text", "number", "unit", ...],
    "notes": ["any footnotes or notes"]
}}

IMPORTANT:
- Preserve all numerical values exactly as shown
- Include units in cells where shown
- Capture all rows and columns
- Use null for empty cells"""

    def __init__(
        self,
        model: Optional[str] = None,
        dpi: int = 150,
    ):
        self.model = model or runtime_config.model_vision
        self.dpi = dpi

    def extract_table(
        self,
        file_path: Path,
        page_num: int,
    ) -> Optional[TableData]:
        """Extract table data from a page."""
        file_path = Path(file_path)
        doc = fitz.open(str(file_path))

        if page_num < 1 or page_num > len(doc):
            doc.close()
            return None

        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=self.dpi)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()
        doc.close()

        client = get_llm_client("vision")

        try:
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": self.TABLE_EXTRACTION_PROMPT, "images": [img_b64]}],
                options={"num_ctx": 8192, "temperature": 0.1},
            )

            return self._parse_table_response(response["message"]["content"], page_num)

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return None

    def _parse_table_response(
        self,
        response_text: str,
        page_num: int,
    ) -> Optional[TableData]:
        """Parse JSON response from vision model."""
        data = parse_json_response(response_text)
        if data is None:
            return None

        try:
            return TableData(
                title=data.get("title", ""),
                source_page=page_num,
                headers=data.get("headers", []),
                rows=data.get("rows", []),
                column_types=data.get("column_types", []),
                notes=data.get("notes", []),
            )
        except Exception:
            return None


# =============================================================================
# Convenience functions
# =============================================================================


def extract_chart(file_path: Path, page_num: int) -> Optional[ChartData]:
    """Convenience function to extract a chart (legacy format)."""
    extractor = GraphExtractor()
    return extractor.extract_chart(file_path, page_num)


def extract_page_charts(file_path: Path, page_num: int) -> Optional[PageChartData]:
    """Convenience function to extract structured chart data from a page."""
    extractor = GraphExtractor()
    return extractor.extract_page_charts(file_path, page_num)


def extract_table(file_path: Path, page_num: int) -> Optional[TableData]:
    """Convenience function to extract a table."""
    extractor = TableExtractor()
    return extractor.extract_table(file_path, page_num)
