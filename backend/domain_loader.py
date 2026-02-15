"""
Domain Pack Loader for ARCA.

Reads ARCA_DOMAIN env var, loads the domain's manifest.json,
and provides accessors for domain-specific configuration.

Usage:
    from domain_loader import get_domain_config

    domain = get_domain_config()
    domain.name          # "example"
    domain.display_name  # "My Domain"
    domain.has_tool("my_tool")  # True
    domain.branding      # {"app_name": "My App", ...}
"""

import json
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve domains directory.
# In Docker: mounted at /app/domains
# Locally: {project_root}/domains
_DOMAINS_DIR = Path(os.environ.get(
    "ARCA_DOMAINS_DIR",
    Path(__file__).resolve().parent.parent / "domains",
))

# Persistent config file for admin-set domain override.
# Stored in data/config/ which is a Docker volume mount.
_CONFIG_DIR = Path(os.environ.get(
    "ARCA_CONFIG_DIR",
    Path(__file__).resolve().parent / "data" / "config",
))
_ACTIVE_DOMAIN_FILE = _CONFIG_DIR / "active_domain.json"


@dataclass
class DomainConfig:
    """Loaded domain pack configuration."""

    name: str
    display_name: str
    version: str
    arca_min_version: str
    description: str
    tools: List[str]
    handlers: List[str]
    routes: List[str]
    lexicon_file: str
    branding: Dict[str, str]
    admin_visible: bool
    domain_dir: Path

    def has_tool(self, tool_name: str) -> bool:
        """Check if this domain provides a tool package."""
        return tool_name in self.tools

    def has_handler(self, handler_name: str) -> bool:
        """Check if this domain provides a handler."""
        return handler_name in self.handlers

    def has_route(self, route_name: str) -> bool:
        """Check if this domain provides a route."""
        return route_name in self.routes

    @property
    def app_name(self) -> str:
        return self.branding.get("app_name", "ARCA")

    @property
    def tagline(self) -> str:
        return self.branding.get("tagline", "")

    @property
    def primary_color(self) -> str:
        return self.branding.get("primary_color", "#6366f1")


# Module-level singleton
_domain_config: Optional[DomainConfig] = None


def _load_domain(domain_name: str) -> DomainConfig:
    """Load a domain pack's manifest.json."""
    domain_dir = _DOMAINS_DIR / domain_name
    manifest_path = domain_dir / "manifest.json"

    if not manifest_path.exists():
        logger.warning(
            f"Domain '{domain_name}' manifest not found at {manifest_path}. "
            f"Falling back to empty config."
        )
        return DomainConfig(
            name=domain_name,
            display_name="ARCA Assistant",
            version="0.0.0",
            arca_min_version="0.1.0",
            description="",
            tools=[],
            handlers=[],
            routes=[],
            lexicon_file="",
            branding={"app_name": "ARCA"},
            admin_visible=True,
            domain_dir=domain_dir,
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    config = DomainConfig(
        name=manifest["name"],
        display_name=manifest.get("display_name", manifest["name"]),
        version=manifest.get("version", "0.0.0"),
        arca_min_version=manifest.get("arca_min_version", "0.1.0"),
        description=manifest.get("description", ""),
        tools=manifest.get("tools", []),
        handlers=manifest.get("handlers", []),
        routes=manifest.get("routes", []),
        lexicon_file=manifest.get("lexicon", ""),
        branding=manifest.get("branding", {}),
        admin_visible=manifest.get("admin_visible", True),
        domain_dir=domain_dir,
    )

    # Add domain tools dir to sys.path so internal tool imports resolve
    # e.g. "from exceedee.checker import ComplianceChecker" works
    tools_dir = domain_dir / "tools"
    if tools_dir.exists() and str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
        logger.debug(f"Added {tools_dir} to sys.path")

    # Add domains parent dir so "from domains.{name}.executors..." works
    domains_parent = _DOMAINS_DIR.parent
    if str(domains_parent) not in sys.path:
        sys.path.insert(0, str(domains_parent))
        logger.debug(f"Added {domains_parent} to sys.path")

    logger.info(
        f"Domain pack loaded: {config.display_name} v{config.version} "
        f"({len(config.tools)} tools, {len(config.handlers)} handlers, "
        f"{len(config.routes)} routes)"
    )
    return config


def _resolve_domain_name() -> str:
    """
    Resolve active domain name.

    Priority: admin config file → ARCA_DOMAIN env var → "example".
    """
    # 1. Check admin-set config file (set via admin panel)
    if _ACTIVE_DOMAIN_FILE.exists():
        try:
            with open(_ACTIVE_DOMAIN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("domain", "").strip()
            if name:
                logger.debug(f"Domain from config file: {name}")
                return name
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read domain config: {e}")

    # 2. Fall back to env var
    return os.environ.get("ARCA_DOMAIN", "example")


def get_domain_config() -> DomainConfig:
    """
    Get the active domain configuration (singleton).

    Resolution: admin config file → ARCA_DOMAIN env var → "example".
    """
    global _domain_config
    if _domain_config is None:
        domain_name = _resolve_domain_name()
        _domain_config = _load_domain(domain_name)
    return _domain_config


def get_domain_name() -> str:
    """Get the active domain name."""
    return get_domain_config().name


def get_domain_tools() -> List[str]:
    """Get list of tool package names provided by the active domain."""
    return get_domain_config().tools


def get_domain_handlers() -> List[str]:
    """Get list of handler names provided by the active domain."""
    return get_domain_config().handlers


def get_domain_routes() -> List[str]:
    """Get list of route names provided by the active domain."""
    return get_domain_config().routes


def get_branding() -> Dict[str, str]:
    """Get branding config for the active domain."""
    return get_domain_config().branding


# Lexicon cache
_lexicon_cache: Optional[Dict] = None


def get_lexicon() -> Dict:
    """
    Get the active domain's lexicon (cached).

    Returns {} if no lexicon file exists.
    """
    global _lexicon_cache
    if _lexicon_cache is not None:
        return _lexicon_cache

    domain = get_domain_config()
    lexicon_path = domain.domain_dir / "lexicon.json"

    if not lexicon_path.exists():
        logger.debug(f"No lexicon for domain '{domain.name}'")
        _lexicon_cache = {}
        return _lexicon_cache

    with open(lexicon_path, "r", encoding="utf-8") as f:
        _lexicon_cache = json.load(f)

    logger.info(f"Loaded lexicon for domain '{domain.name}' ({len(_lexicon_cache)} keys)")
    return _lexicon_cache


# Pipeline defaults — vanilla ARCA, no domain expertise
_PIPELINE_DEFAULTS = {
    "specialty": "scientific and engineering disciplines",
    "reference_type": "a technical reference document",
    "raptor_context": "technical documentation",
    "raptor_summary_intro": "Summarize the following technical content.",
    "raptor_preserve": [
        "Key terminology and definitions",
        "Equations and formulas",
        "Numerical values and specifications",
        "Standards and regulatory references",
    ],
    "confidence_example": "empirical correlation, limited sample data in project area",
    "equation_example": "$$F = ma$$, $$E = mc^2$$, $$\\sigma = \\frac{F}{A}$$",
    "default_topic": "general",
}


def get_pipeline_config() -> Dict:
    """Get pipeline config from lexicon with generic fallbacks.

    Domain packs inject expertise via lexicon.pipeline section.
    Vanilla ARCA gets neutral defaults suitable for any domain.
    """
    lexicon = get_lexicon()
    pipeline = lexicon.get("pipeline", {})
    result = {k: pipeline.get(k, v) for k, v in _PIPELINE_DEFAULTS.items()}
    # Also include domain-specific pipeline keys not in defaults
    for k, v in pipeline.items():
        if k not in result:
            result[k] = v
    return result


def list_available_domains() -> List[Dict]:
    """List all domain packs with manifest info."""
    domains = []
    if not _DOMAINS_DIR.exists():
        return domains

    for entry in sorted(_DOMAINS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                domains.append({
                    "name": manifest.get("name", entry.name),
                    "display_name": manifest.get("display_name", entry.name),
                    "description": manifest.get("description", ""),
                    "version": manifest.get("version", "0.0.0"),
                    "tools_count": len(manifest.get("tools", [])),
                    "routes_count": len(manifest.get("routes", [])),
                    "admin_visible": manifest.get("admin_visible", True),
                })
            except (json.JSONDecodeError, OSError):
                domains.append({
                    "name": entry.name,
                    "display_name": entry.name,
                    "description": "Error reading manifest",
                    "version": "?",
                    "tools_count": 0,
                    "routes_count": 0,
                })

    return domains


def set_active_domain(name: str) -> None:
    """Set the active domain via config file (persists across restarts)."""
    # Validate domain exists
    domain_dir = _DOMAINS_DIR / name
    if not (domain_dir / "manifest.json").exists():
        raise ValueError(f"Domain '{name}' not found in {_DOMAINS_DIR}")

    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ACTIVE_DOMAIN_FILE, "w", encoding="utf-8") as f:
        json.dump({"domain": name}, f)
    logger.info(f"Active domain set to '{name}' (config file)")


def reload_domain() -> DomainConfig:
    """Clear caches and reload domain configuration."""
    global _domain_config, _lexicon_cache
    _domain_config = None
    _lexicon_cache = None
    return get_domain_config()
