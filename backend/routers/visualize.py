"""
Visualize Router - 3D Visualization Data API

Endpoints:
- POST /api/viz/create - Create new visualization, returns ID
- GET /api/viz/data/{viz_id} - Returns Plotly JSON data
- GET /api/viz/preview/{viz_id}.png - Returns PNG preview image
- GET /api/viz/metadata/{viz_id} - Returns metadata (borehole count, etc.)
"""

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/viz", tags=["visualize"])

# In-memory store for visualization data
_viz_store: Dict[str, Dict] = {}

# Directory for storing preview images
BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports"

# Cleanup threshold
VIZ_MAX_AGE_HOURS = 1


class VisualizationCreate(BaseModel):
    """Request to create a new visualization."""

    plotly_data: Dict[str, Any]
    preview_base64: Optional[str] = None  # Base64-encoded PNG
    metadata: Optional[Dict[str, Any]] = None


class VisualizationResponse(BaseModel):
    """Response after creating a visualization."""

    viz_id: str
    created_at: str
    preview_url: Optional[str] = None


def _cleanup_old_visualizations() -> int:
    """
    Remove visualizations older than VIZ_MAX_AGE_HOURS.

    Returns:
        Number of visualizations cleaned up.
    """
    cutoff = datetime.now() - timedelta(hours=VIZ_MAX_AGE_HOURS)
    to_delete = []

    for viz_id, viz_data in _viz_store.items():
        created_at = viz_data.get("created_at")
        if isinstance(created_at, datetime) and created_at < cutoff:
            to_delete.append(viz_id)

    for viz_id in to_delete:
        # Remove from store
        viz_data = _viz_store.pop(viz_id, None)

        # Remove preview file if exists
        if viz_data:
            preview_path = viz_data.get("preview_path")
            if preview_path:
                try:
                    Path(preview_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete preview for {viz_id}: {e}")

        logger.debug(f"Cleaned up old visualization: {viz_id}")

    if to_delete:
        logger.info(f"Cleaned up {len(to_delete)} old visualizations")

    return len(to_delete)


def _save_preview_image(viz_id: str, preview_base64: str) -> Optional[Path]:
    """
    Save base64-encoded PNG preview to disk.

    Args:
        viz_id: Visualization ID
        preview_base64: Base64-encoded PNG data

    Returns:
        Path to saved file, or None if failed
    """
    import base64

    try:
        # Ensure reports directory exists
        REPORTS_DIR.mkdir(exist_ok=True)

        # Decode base64
        # Handle data URL format: "data:image/png;base64,..."
        if "," in preview_base64:
            preview_base64 = preview_base64.split(",", 1)[1]

        image_data = base64.b64decode(preview_base64)

        # Save to file
        preview_path = REPORTS_DIR / f"viz_{viz_id}.png"
        preview_path.write_bytes(image_data)

        return preview_path

    except Exception as e:
        logger.error(f"Failed to save preview image for {viz_id}: {e}")
        return None


@router.post("/create", response_model=VisualizationResponse)
async def create_visualization(data: VisualizationCreate) -> VisualizationResponse:
    """
    Create a new visualization.

    Stores Plotly figure data and optional PNG preview.
    Returns a viz_id for later retrieval.
    """
    # Cleanup old visualizations first
    _cleanup_old_visualizations()

    # Generate unique ID
    viz_id = str(uuid.uuid4())[:8]
    created_at = datetime.now()

    # Save preview image if provided
    preview_path = None
    preview_url = None
    if data.preview_base64:
        preview_path = _save_preview_image(viz_id, data.preview_base64)
        if preview_path:
            preview_url = f"/api/viz/preview/{viz_id}.png"

    # Store visualization data
    _viz_store[viz_id] = {
        "plotly_data": data.plotly_data,
        "preview_path": str(preview_path) if preview_path else None,
        "created_at": created_at,
        "metadata": data.metadata or {},
    }

    logger.info(f"Created visualization {viz_id}")

    return VisualizationResponse(
        viz_id=viz_id,
        created_at=created_at.isoformat(),
        preview_url=preview_url,
    )


@router.get("/data/{viz_id}")
async def get_visualization_data(viz_id: str) -> Dict[str, Any]:
    """
    Get Plotly JSON data for a visualization.

    Returns the full Plotly figure data for rendering.
    """
    viz_data = _viz_store.get(viz_id)

    if not viz_data:
        raise HTTPException(status_code=404, detail=f"Visualization not found: {viz_id}")

    return {
        "viz_id": viz_id,
        "plotly_data": viz_data["plotly_data"],
    }


@router.get("/preview/{viz_id}.png")
async def get_visualization_preview(viz_id: str) -> FileResponse:
    """
    Get PNG preview image for a visualization.

    Returns the preview image file.
    """
    viz_data = _viz_store.get(viz_id)

    if not viz_data:
        raise HTTPException(status_code=404, detail=f"Visualization not found: {viz_id}")

    preview_path = viz_data.get("preview_path")

    if not preview_path:
        raise HTTPException(status_code=404, detail=f"No preview available for: {viz_id}")

    path = Path(preview_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Preview file not found: {viz_id}")

    # Security: ensure path is within REPORTS_DIR
    try:
        path.resolve().relative_to(REPORTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=path,
        media_type="image/png",
        filename=f"viz_{viz_id}.png",
    )


@router.get("/metadata/{viz_id}")
async def get_visualization_metadata(viz_id: str) -> Dict[str, Any]:
    """
    Get metadata for a visualization.

    Returns metadata like borehole count, section info, etc.
    """
    viz_data = _viz_store.get(viz_id)

    if not viz_data:
        raise HTTPException(status_code=404, detail=f"Visualization not found: {viz_id}")

    created_at = viz_data["created_at"]

    return {
        "viz_id": viz_id,
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
        "metadata": viz_data.get("metadata", {}),
        "has_preview": viz_data.get("preview_path") is not None,
    }


@router.delete("/{viz_id}")
async def delete_visualization(viz_id: str) -> Dict[str, Any]:
    """
    Delete a visualization.

    Removes the visualization data and preview file.
    """
    viz_data = _viz_store.pop(viz_id, None)

    if not viz_data:
        raise HTTPException(status_code=404, detail=f"Visualization not found: {viz_id}")

    # Remove preview file if exists
    preview_path = viz_data.get("preview_path")
    if preview_path:
        try:
            Path(preview_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete preview for {viz_id}: {e}")

    logger.info(f"Deleted visualization {viz_id}")

    return {"success": True, "viz_id": viz_id}


@router.post("/generate/3d")
async def generate_3d_from_cache() -> Dict[str, Any]:
    """
    Generate 3D visualization from cached borehole data.

    This endpoint allows direct generation without going through chat.
    Uses the borehole data cached from the last extract_borehole_log call.
    """
    from routers.chat_executors.documents import _extracted_boreholes

    # Get extracted boreholes
    logs = _extracted_boreholes.get("latest")
    if not logs:
        raise HTTPException(status_code=400, detail="No borehole data available. Extract borehole logs first.")

    # Filter to logs with coordinates
    logs_with_coords = [l for l in logs if l.northing and l.easting]
    if len(logs_with_coords) < 2:
        raise HTTPException(
            status_code=400, detail=f"Need at least 2 boreholes with coordinates, found {len(logs_with_coords)}"
        )

    try:
        from tools.loggview.viz3d import BoreholeViz3D
    except ImportError:
        raise HTTPException(status_code=501, detail="Visualization not available — requires domain pack with borehole tools")

    # Generate visualization
    viz = BoreholeViz3D(logs_with_coords)
    plotly_data = viz.generate_figure()

    # Generate unique ID and store
    viz_id = str(uuid.uuid4())[:8]
    created_at = datetime.now()

    # Try to generate preview
    preview_path = None
    preview_url = None
    try:
        REPORTS_DIR.mkdir(exist_ok=True)
        preview_file = REPORTS_DIR / f"viz_{viz_id}.png"
        viz.generate_preview(str(preview_file), width=800, height=600)
        preview_path = str(preview_file)
        preview_url = f"/api/viz/preview/{viz_id}.png"
    except Exception as e:
        logger.warning(f"Could not generate preview: {e}")

    # Store visualization data
    _viz_store[viz_id] = {
        "plotly_data": plotly_data,
        "preview_path": preview_path,
        "created_at": created_at,
        "metadata": {
            "borehole_count": len(logs_with_coords),
            "title": "3D Borehole Visualization",
        },
    }

    logger.info(f"Generated 3D visualization {viz_id} with {len(logs_with_coords)} boreholes")

    return {
        "success": True,
        "viz_id": viz_id,
        "preview_url": preview_url,
        "data_url": f"/api/viz/data/{viz_id}",
        "fullpage_url": f"/viz/3d/{viz_id}",
        "borehole_count": len(logs_with_coords),
    }


@router.post("/generate/cross-section")
async def generate_cross_section_from_cache(section_name: str = "Section A-A'") -> Dict[str, Any]:
    """
    Generate cross-section from cached borehole data.

    This endpoint allows direct generation without going through chat.
    """
    from routers.chat_executors.documents import _extracted_boreholes

    # Get extracted boreholes
    logs = _extracted_boreholes.get("latest")
    if not logs:
        raise HTTPException(status_code=400, detail="No borehole data available. Extract borehole logs first.")

    # Filter to logs with coordinates
    logs_with_coords = [l for l in logs if l.northing and l.easting]
    if len(logs_with_coords) < 2:
        raise HTTPException(
            status_code=400, detail=f"Need at least 2 boreholes with coordinates, found {len(logs_with_coords)}"
        )

    try:
        from tools.loggview.cross_section import CrossSectionGenerator
    except ImportError:
        raise HTTPException(status_code=501, detail="Visualization not available — requires domain pack with borehole tools")

    # Generate cross-section
    generator = CrossSectionGenerator(logs_with_coords)
    section = generator.auto_section(name=section_name)

    # Save to reports directory
    REPORTS_DIR.mkdir(exist_ok=True)
    output_name = "cross_section.png"
    output_path = REPORTS_DIR / output_name

    generator.save(
        str(output_path),
        section=section,
        title=section_name,
        show_water=True,
        show_samples=True,
        dpi=150,
    )

    download_path = f"/api/download/{output_name}"

    logger.info(f"Generated cross-section: {section_name}, {len(logs_with_coords)} boreholes")

    return {
        "success": True,
        "cross_section_file": download_path,
        "section_name": section_name,
        "section_length": f"{section.length:.0f}m",
        "section_bearing": f"{section.bearing:.0f}°",
        "boreholes_included": len(logs_with_coords),
    }


@router.get("/list")
async def list_visualizations() -> Dict[str, Any]:
    """
    List all stored visualizations.

    Returns summary info for each visualization.
    """
    # Cleanup old visualizations first
    _cleanup_old_visualizations()

    visualizations = []

    for viz_id, viz_data in _viz_store.items():
        created_at = viz_data["created_at"]
        visualizations.append(
            {
                "viz_id": viz_id,
                "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
                "metadata": viz_data.get("metadata", {}),
                "has_preview": viz_data.get("preview_path") is not None,
            }
        )

    return {
        "count": len(visualizations),
        "visualizations": visualizations,
    }
