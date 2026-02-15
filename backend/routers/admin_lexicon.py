"""
Admin Lexicon Router - Domain Lexicon Management

Provides CRUD endpoints for the active domain's lexicon.json:
- GET/PUT the full lexicon
- POST/DELETE individual terminology terms

Protected by admin authentication.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from domain_loader import get_domain_config, get_lexicon, reload_domain
from routers.chat_prompts import clear_prompt_caches
from services.admin_auth import verify_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/lexicon", tags=["admin-lexicon"])


def _lexicon_path() -> Path:
    """Get the filesystem path to the active domain's lexicon.json."""
    domain = get_domain_config()
    return domain.domain_dir / "lexicon.json"


def _read_lexicon() -> Dict[str, Any]:
    """Read lexicon.json from disk (bypasses cache for freshness)."""
    path = _lexicon_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="No lexicon.json for active domain")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_lexicon(data: Dict[str, Any]) -> None:
    """Write lexicon.json to disk, then reload domain caches."""
    path = _lexicon_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    # Reload domain config + lexicon cache and clear prompt caches
    reload_domain()
    clear_prompt_caches()
    logger.info("Lexicon updated and caches refreshed")


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("", dependencies=[Depends(verify_admin)])
async def get_lexicon_data():
    """Return the full lexicon.json for the active domain."""
    data = _read_lexicon()
    domain = get_domain_config()
    return {
        "domain": domain.name,
        "display_name": domain.display_name,
        "lexicon": data,
    }


class LexiconUpdate(BaseModel):
    lexicon: Dict[str, Any]


@router.put("", dependencies=[Depends(verify_admin)])
async def update_lexicon(body: LexiconUpdate):
    """Replace the full lexicon.json content."""
    _write_lexicon(body.lexicon)
    return {"status": "ok", "message": "Lexicon saved"}


class TermEntry(BaseModel):
    term: str
    variants: Any  # list of [pattern, display] pairs or list of strings


@router.post("/term", dependencies=[Depends(verify_admin)])
async def add_term(body: TermEntry):
    """Add or update a single terminology_variants entry."""
    data = _read_lexicon()
    if "terminology_variants" not in data:
        data["terminology_variants"] = {}
    data["terminology_variants"][body.term] = body.variants
    _write_lexicon(data)
    return {"status": "ok", "term": body.term}


@router.delete("/term/{term}", dependencies=[Depends(verify_admin)])
async def delete_term(term: str):
    """Remove a terminology_variants entry."""
    data = _read_lexicon()
    variants = data.get("terminology_variants", {})
    if term not in variants:
        raise HTTPException(status_code=404, detail=f"Term '{term}' not found")
    del variants[term]
    _write_lexicon(data)
    return {"status": "ok", "deleted": term}
