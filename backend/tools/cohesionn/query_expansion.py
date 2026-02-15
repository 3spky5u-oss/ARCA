"""
Cohesionn Query Expansion - Engineering-aware synonym expansion

Expands queries with domain-specific synonyms to improve recall.
Example: "pile load test" -> "pile load test static load test deep foundation"

Built from TOPIC_KEYWORDS bidirectional mappings + engineering thesaurus.
"""

import logging
from typing import List, Set, Dict, Optional

logger = logging.getLogger(__name__)


# Synonym groups loaded from lexicon pipeline config
_synonym_cache = None
_expansion_dict_cache = None


def _get_synonym_groups():
    """Load synonym groups from lexicon pipeline config."""
    global _synonym_cache
    if _synonym_cache is not None:
        return _synonym_cache
    try:
        from domain_loader import get_pipeline_config
        pipeline = get_pipeline_config()
        raw = pipeline.get("rag_synonyms", [])
        # Convert lists back to sets (JSON doesn't support sets)
        _synonym_cache = [set(group) for group in raw] if raw else []
    except Exception:
        _synonym_cache = []
    return _synonym_cache


def clear_synonym_cache():
    """Clear synonym and expansion dict caches (call after domain/lexicon change)."""
    global _synonym_cache, _expansion_dict_cache
    _synonym_cache = None
    _expansion_dict_cache = None


def _build_expansion_dict() -> Dict[str, Set[str]]:
    """Build bidirectional expansion dictionary from synonym groups"""
    expansion_dict: Dict[str, Set[str]] = {}

    for group in _get_synonym_groups():
        for term in group:
            term_lower = term.lower()
            if term_lower not in expansion_dict:
                expansion_dict[term_lower] = set()
            for other in group:
                other_lower = other.lower()
                if other_lower != term_lower:
                    expansion_dict[term_lower].add(other_lower)

    return expansion_dict


def _get_expansion_dict() -> Dict[str, Set[str]]:
    """Get cached expansion dictionary, building if needed."""
    global _expansion_dict_cache
    if _expansion_dict_cache is None:
        _expansion_dict_cache = _build_expansion_dict()
    return _expansion_dict_cache


def expand_query(query: str, max_expansions: int = 3) -> str:
    """
    Expand query with engineering synonyms.

    Args:
        query: Original query
        max_expansions: Maximum synonym additions per matched term

    Returns:
        Expanded query with synonyms appended
    """
    query_lower = query.lower()
    expansions: Set[str] = set()

    # Check for exact phrase matches first (longer phrases)
    for term, synonyms in sorted(_get_expansion_dict().items(), key=lambda x: -len(x[0])):
        if term in query_lower:
            # Add up to max_expansions synonyms
            for i, syn in enumerate(sorted(synonyms, key=len)):
                if i >= max_expansions:
                    break
                # Don't add if already in query
                if syn not in query_lower:
                    expansions.add(syn)

    if expansions:
        expanded = f"{query} {' '.join(expansions)}"
        logger.debug(f"Query expanded: '{query}' -> '{expanded}'")
        return expanded

    return query


def get_synonyms(term: str) -> Set[str]:
    """Get synonyms for a single term"""
    return _get_expansion_dict().get(term.lower(), set())


def expand_terms(terms: List[str]) -> List[str]:
    """
    Expand a list of terms with their synonyms.

    Useful for tokenized queries.

    Args:
        terms: List of query terms

    Returns:
        Expanded list including synonyms
    """
    expanded = set(terms)

    for term in terms:
        synonyms = get_synonyms(term)
        expanded.update(synonyms)

    return list(expanded)


class QueryExpander:
    """
    Query expansion with configuration control.

    Wraps the module functions with config-based enable/disable.
    """

    def __init__(self, enabled: bool = True, max_expansions: int = 3):
        """
        Args:
            enabled: Whether expansion is enabled
            max_expansions: Max synonyms per matched term
        """
        self.enabled = enabled
        self.max_expansions = max_expansions

    def expand(self, query: str) -> str:
        """Expand query if enabled"""
        if not self.enabled:
            return query
        return expand_query(query, self.max_expansions)

    @classmethod
    def from_config(cls) -> "QueryExpander":
        """Create expander from runtime config"""
        from config import runtime_config
        return cls(enabled=runtime_config.query_expansion_enabled)


# Module-level convenience
_expander: Optional[QueryExpander] = None


def get_query_expander() -> QueryExpander:
    """Get singleton query expander"""
    global _expander
    if _expander is None:
        _expander = QueryExpander.from_config()
    return _expander
