"""
Query Classifier - Classify queries as global or local

Routes queries to appropriate retrieval strategy:
- GLOBAL: Broad theme questions -> Community search
- LOCAL: Specific factual questions -> Standard chunk retrieval
- HYBRID: Questions that benefit from both
"""

import logging
import re
from typing import List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


def _get_local_patterns() -> List[str]:
    """Load domain-specific LOCAL_PATTERNS from lexicon pipeline config."""
    try:
        from domain_loader import get_pipeline_config
        return get_pipeline_config().get("query_local_patterns", [])
    except Exception:
        return []


class QueryType(Enum):
    """Query classification types."""

    GLOBAL = "global"     # Broad/conceptual -> Community search
    LOCAL = "local"       # Specific/factual -> Standard retrieval
    HYBRID = "hybrid"     # Both strategies
    CROSS_REF = "cross_ref"  # Cross-reference -> GraphRAG traversal


class QueryClassifier:
    """
    Classify queries to determine retrieval strategy.

    Uses pattern matching and heuristics:
    - Global: Overview questions, comparisons, broad themes
    - Local: Specific values, calculations, standard references
    - Hybrid: Questions that could benefit from both

    Fast pattern-based classification (no LLM call needed).
    """

    # Patterns indicating CROSS-REFERENCE queries (GraphRAG: +0.024 composite)
    CROSS_REF_PATTERNS = [
        r"compare (?:and contrast)?",
        r"difference(?:s)? between",
        r"(?:vs|versus)",
        r"similarities between",
        r"how (?:does|do) .+ (?:differ|compare)",
    ]

    # Patterns indicating GLOBAL queries (broad/conceptual)
    GLOBAL_PATTERNS = [
        # Overview questions
        r"what are the (?:main|major|key|primary|different|various)",
        r"(?:overview|summary|introduction|fundamentals) of",
        r"explain (?:the|how|what|why)",
        r"describe (?:the|how|what)",
        r"types? of",
        r"categories? of",
        r"classification of",
        # Comparison (broad, non-cross-ref)
        r"pros and cons",
        r"advantages (?:and disadvantages)?",
        # Broad scope
        r"(?:general|overall|broad) (?:considerations|approach|guidelines)",
        r"considerations? for",
        r"factors? (?:affecting|influencing|that)",
        r"principles? of",
        r"best practices? for",
        # Theme questions
        r"how do(?:es)? .+ work",
        r"what is (?:the role|importance) of",
        r"when (?:should|to) use",
    ]

    # Core patterns indicating LOCAL queries (specific/factual)
    _CORE_LOCAL_PATTERNS = [
        # Specific values
        r"what is the (?:value|range|typical|exact)",
        r"how much",
        r"how many",
        r"what (?:is|are) the (?:specific|exact)",
        # Calculations
        r"calculate",
        r"formula for",
        r"equation for",
        r"compute",
        # Standard references
        r"according to (?:astm|aashto|asce|csa)",
        r"(?:astm|aashto|asce) [a-z]?\d+",
        r"per (?:astm|aashto|code)",
        r"specification for",
        # Lookup questions
        r"define ",
        r"definition of",
    ]

    @classmethod
    def _build_local_patterns(cls):
        """Combine core LOCAL patterns with domain-specific ones from lexicon."""
        return cls._CORE_LOCAL_PATTERNS + _get_local_patterns()

    def __init__(self, default_type: QueryType = QueryType.HYBRID):
        """
        Args:
            default_type: Default classification when patterns don't match
        """
        self.default_type = default_type

        # Compile patterns for efficiency
        self._cross_ref_regex = [re.compile(p, re.IGNORECASE) for p in self.CROSS_REF_PATTERNS]
        self._global_regex = [re.compile(p, re.IGNORECASE) for p in self.GLOBAL_PATTERNS]
        local_patterns = self._build_local_patterns()
        self._local_regex = [re.compile(p, re.IGNORECASE) for p in local_patterns]

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify a query.

        Args:
            query: The search query

        Returns:
            Tuple of (QueryType, confidence score 0-1)
        """
        query_lower = query.lower().strip()

        # Check cross-reference patterns first (highest priority)
        cross_ref_matches = sum(1 for p in self._cross_ref_regex if p.search(query_lower))
        if cross_ref_matches > 0:
            confidence = min(0.95, 0.5 + cross_ref_matches * 0.2)
            return QueryType.CROSS_REF, confidence

        # Count pattern matches
        global_matches = sum(1 for p in self._global_regex if p.search(query_lower))
        local_matches = sum(1 for p in self._local_regex if p.search(query_lower))

        # Calculate confidence based on match counts
        total_matches = global_matches + local_matches

        if total_matches == 0:
            # Use heuristics for unmatched queries
            return self._heuristic_classify(query), 0.5

        # Determine type based on matches
        if global_matches > local_matches:
            confidence = global_matches / (global_matches + local_matches + 1)
            return QueryType.GLOBAL, min(0.95, confidence + 0.3)

        elif local_matches > global_matches:
            confidence = local_matches / (global_matches + local_matches + 1)
            return QueryType.LOCAL, min(0.95, confidence + 0.3)

        else:
            # Equal matches -> HYBRID
            return QueryType.HYBRID, 0.6

    def _heuristic_classify(self, query: str) -> QueryType:
        """Heuristic classification for unmatched queries."""
        query_lower = query.lower()
        word_count = len(query.split())

        # Very short queries are often broad
        if word_count <= 3:
            return QueryType.HYBRID

        # Questions with "what are" tend to be broader
        if query_lower.startswith(("what are", "how do", "why")):
            return QueryType.GLOBAL

        # Questions with "what is" + specific term tend to be local
        if query_lower.startswith("what is") and word_count <= 6:
            return QueryType.LOCAL

        # Long specific questions tend to be local
        if word_count >= 10:
            return QueryType.LOCAL

        return self.default_type

    def should_use_graph_search(self, query: str) -> bool:
        """Quick check if GraphRAG should be activated for cross-reference queries."""
        query_type, confidence = self.classify(query)
        return query_type == QueryType.CROSS_REF and confidence >= 0.5

    def should_use_global_search(self, query: str) -> bool:
        """Quick check if global search should be used."""
        query_type, confidence = self.classify(query)
        return query_type in (QueryType.GLOBAL, QueryType.HYBRID) and confidence >= 0.5

    def should_use_local_search(self, query: str) -> bool:
        """Quick check if local search should be used."""
        query_type, confidence = self.classify(query)
        return query_type in (QueryType.LOCAL, QueryType.HYBRID) and confidence >= 0.5

    def get_search_strategy(self, query: str) -> dict:
        """
        Get recommended search strategy.

        Returns:
            Dict with flags for each search type
        """
        query_type, confidence = self.classify(query)

        return {
            "query_type": query_type.value,
            "confidence": confidence,
            "use_community_search": query_type in (QueryType.GLOBAL, QueryType.HYBRID),
            "use_chunk_search": query_type in (QueryType.LOCAL, QueryType.HYBRID, QueryType.CROSS_REF),
            "use_graph_search": query_type == QueryType.CROSS_REF,
            "community_weight": 0.6 if query_type == QueryType.GLOBAL else 0.3,
        }


# Singleton
_query_classifier = None


def get_query_classifier() -> QueryClassifier:
    """Get or create singleton classifier."""
    global _query_classifier

    if _query_classifier is None:
        _query_classifier = QueryClassifier()

    return _query_classifier
