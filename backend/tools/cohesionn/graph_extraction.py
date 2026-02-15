"""
GraphRAG Entity Extraction - Extract entities and relationships from chunks

Uses SpaCy for pattern matching and LLM for relationship extraction.
Domain-configurable: core extracts Standards only; domain packs add
custom entity types and parameters via lexicon pipeline config.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """An extracted entity."""

    name: str
    entity_type: str  # Standard, TestMethod, SoilType, Parameter, Equipment
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunk_id: Optional[str] = None


@dataclass
class Relationship:
    """A relationship between entities."""

    source: str  # Entity name
    target: str  # Entity name
    rel_type: str  # REFERENCES, APPLIES_TO, MEASURES, REQUIRES, CONTAINS
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result from entity extraction."""

    entities: List[Entity]
    relationships: List[Relationship]
    chunk_id: Optional[str] = None


# =============================================================================
# Pattern-based Entity Extraction
# =============================================================================

# Core patterns — cross-domain, always active
STANDARD_PATTERNS = [
    # ASTM D1586-22 or ASTM D 1586
    r"ASTM\s*D?\s*(\d{3,5})(?:-\d{2,4})?",
    # AASHTO T99-22 or AASHTO T 99
    r"AASHTO\s*([TM]\s*\d{2,4})(?:-\d{2,4})?",
    # ASCE 7-22
    r"ASCE\s*(\d+)(?:-\d{2,4})?",
    # CSA standards
    r"CSA\s*([A-Z]?\d+\.?\d*)(?:-\d{2,4})?",
    # ISO standards
    r"ISO\s*(\d{4,5})(?:-\d)?(?::\d{4})?",
]

# =============================================================================
# Domain-configurable patterns (loaded from lexicon pipeline config)
# =============================================================================

_domain_patterns_cache = None


def _get_domain_patterns():
    """Load domain-specific entity patterns from lexicon pipeline config."""
    global _domain_patterns_cache
    if _domain_patterns_cache is not None:
        return _domain_patterns_cache
    try:
        from domain_loader import get_pipeline_config
        pipeline = get_pipeline_config()
        # Entity type names are defined by the domain pack's lexicon.json pipeline config.
        # Default keys (soil_types, test_methods, etc.) are defined by the active domain pack.
        _domain_patterns_cache = {
            "test_methods": pipeline.get("graph_test_methods", {}),
            "soil_types": pipeline.get("graph_soil_types", {}),
            "parameters": pipeline.get("graph_parameters", {}),
            "equipment": pipeline.get("graph_equipment", {}),
        }
    except Exception:
        _domain_patterns_cache = {"test_methods": {}, "soil_types": {}, "parameters": {}, "equipment": {}}
    return _domain_patterns_cache


def clear_domain_patterns_cache():
    """Clear cached domain patterns (called on domain switch)."""
    global _domain_patterns_cache
    _domain_patterns_cache = None


class EntityExtractor:
    """
    Extract engineering entities from text using pattern matching.

    Stage 1: Pattern matching for known terms (fast, high precision)
    Stage 2: Optional LLM refinement for relationships (slower, better recall)
    """

    def __init__(self, use_spacy: bool = True, use_llm: bool = False):
        """
        Args:
            use_spacy: Use SpaCy for additional NER (if available)
            use_llm: Use LLM for relationship extraction
        """
        self.use_spacy = use_spacy
        self.use_llm = use_llm
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load SpaCy model."""
        if self._nlp is None and self.use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy model loaded for entity extraction")
            except ImportError:
                logger.warning("SpaCy not installed, using pattern-only extraction")
                self.use_spacy = False
            except OSError:
                logger.warning("SpaCy model not found, using pattern-only extraction")
                self.use_spacy = False

        return self._nlp

    def extract(
        self,
        text: str,
        chunk_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from text.

        Args:
            text: Document text
            chunk_id: Optional chunk ID for provenance

        Returns:
            ExtractionResult with entities and relationships
        """
        entities = []
        relationships = []

        # Stage 1: Pattern-based extraction
        # Core: Standards (always active, cross-domain)
        entities.extend(self._extract_standards(text, chunk_id))

        # Domain-specific: only if patterns configured in lexicon pipeline
        domain = _get_domain_patterns()
        if domain["test_methods"]:
            entities.extend(self._extract_test_methods(text, chunk_id, domain["test_methods"]))
        if domain["soil_types"]:
            entities.extend(self._extract_soil_types(text, chunk_id, domain["soil_types"]))
        if domain["parameters"]:
            entities.extend(self._extract_parameters(text, chunk_id, domain["parameters"]))
        if domain["equipment"]:
            entities.extend(self._extract_equipment(text, chunk_id, domain["equipment"]))

        # Stage 2: SpaCy NER for additional entities
        if self.use_spacy and self.nlp:
            entities.extend(self._extract_with_spacy(text, chunk_id))

        # Stage 3: Infer relationships from co-occurrence
        relationships.extend(self._infer_relationships(entities, text))

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            chunk_id=chunk_id,
        )

    def _extract_standards(
        self,
        text: str,
        chunk_id: Optional[str],
    ) -> List[Entity]:
        """Extract standard references (ASTM, AASHTO, etc.)."""
        entities = []

        for pattern in STANDARD_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                full_match = match.group(0).upper()

                # Determine organization
                if "ASTM" in full_match:
                    org = "ASTM"
                elif "AASHTO" in full_match:
                    org = "AASHTO"
                elif "ASCE" in full_match:
                    org = "ASCE"
                elif "CSA" in full_match:
                    org = "CSA"
                elif "ISO" in full_match:
                    org = "ISO"
                else:
                    org = "Unknown"

                # Normalize code
                code = re.sub(r"\s+", " ", full_match).strip()

                entities.append(
                    Entity(
                        name=code,
                        entity_type="Standard",
                        confidence=0.95,
                        properties={"org": org, "code": code},
                        source_chunk_id=chunk_id,
                    )
                )

        return entities

    def _extract_test_methods(
        self,
        text: str,
        chunk_id: Optional[str],
        patterns: Dict[str, Any] = None,
    ) -> List[Entity]:
        """Extract test method references using domain-provided patterns.

        Args:
            patterns: Dict mapping regex -> [abbrev, full_name] from lexicon
        """
        entities = []
        if not patterns:
            return entities

        for pattern, value in patterns.items():
            # Normalize: lexicon JSON gives lists, code used tuples
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                abbrev, full_name = value[0], value[1]
            else:
                continue
            if re.search(pattern, text):
                entities.append(
                    Entity(
                        name=abbrev,
                        entity_type="TestMethod",
                        confidence=0.9,
                        properties={
                            "abbreviation": abbrev,
                            "full_name": full_name,
                        },
                        source_chunk_id=chunk_id,
                    )
                )

        return entities

    def _extract_soil_types(
        self,
        text: str,
        chunk_id: Optional[str],
        patterns: Dict[str, str] = None,
    ) -> List[Entity]:
        """Extract domain classification codes using domain-provided patterns.

        Args:
            patterns: Dict mapping regex -> description string from lexicon
        """
        entities = []
        if not patterns:
            return entities

        for pattern, description in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append(
                    Entity(
                        name=match,
                        entity_type="SoilType",
                        confidence=0.85,
                        properties={
                            "classification": match,
                            "description": description,
                        },
                        source_chunk_id=chunk_id,
                    )
                )

        return entities

    def _extract_parameters(
        self,
        text: str,
        chunk_id: Optional[str],
        patterns: Dict[str, Any] = None,
    ) -> List[Entity]:
        """Extract engineering parameters using domain-provided patterns.

        Args:
            patterns: Dict mapping regex -> [symbol, description, unit] from lexicon
        """
        entities = []
        if not patterns:
            return entities

        for pattern, value in patterns.items():
            # Normalize: lexicon JSON gives lists, code used tuples
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                symbol, description, unit = value[0], value[1], value[2]
            else:
                continue
            if re.search(pattern, text):
                entities.append(
                    Entity(
                        name=symbol,
                        entity_type="Parameter",
                        confidence=0.85,
                        properties={
                            "symbol": symbol,
                            "description": description,
                            "unit": unit,
                        },
                        source_chunk_id=chunk_id,
                    )
                )

        return entities

    def _extract_equipment(
        self,
        text: str,
        chunk_id: Optional[str],
        patterns: Dict[str, str] = None,
    ) -> List[Entity]:
        """Extract equipment entities using domain-provided patterns.

        Args:
            patterns: Dict mapping regex -> equipment name from lexicon
        """
        entities = []
        if not patterns:
            return entities

        for pattern, name in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities.append(
                    Entity(
                        name=name,
                        entity_type="Equipment",
                        confidence=0.8,
                        properties={},
                        source_chunk_id=chunk_id,
                    )
                )

        return entities

    # SpaCy label -> our entity type. Labels not listed here are ignored.
    _SPACY_LABEL_MAP = {
        "ORG": "Organization",
        "PERSON": "Person",
        "GPE": "Location",
        "LOC": "Location",
        "PRODUCT": "Equipment",
        "LAW": "Standard",
    }

    # Too generic to be useful as graph nodes
    _SPACY_SKIP = {"DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"}

    def _extract_with_spacy(
        self,
        text: str,
        chunk_id: Optional[str],
    ) -> List[Entity]:
        """Extract additional entities using SpaCy NER."""
        entities = []

        if not self.nlp:
            return entities

        doc = self.nlp(text[:10000])  # Limit to avoid memory issues

        for ent in doc.ents:
            if ent.label_ in self._SPACY_SKIP:
                continue

            # Skip standards orgs already captured by pattern extraction
            if ent.label_ == "ORG" and any(
                org in ent.text.upper() for org in ["ASTM", "AASHTO", "ASCE", "CSA", "ISO"]
            ):
                continue

            # Skip very short entities (single chars, initials)
            if len(ent.text.strip()) < 3:
                continue

            entity_type = self._SPACY_LABEL_MAP.get(ent.label_)
            if not entity_type:
                continue

            entities.append(
                Entity(
                    name=ent.text.strip(),
                    entity_type=entity_type,
                    confidence=0.6,
                    properties={"spacy_label": ent.label_},
                    source_chunk_id=chunk_id,
                )
            )

        return entities

    def _infer_relationships(
        self,
        entities: List[Entity],
        text: str,
    ) -> List[Relationship]:
        """Infer relationships from entity co-occurrence."""
        relationships = []

        # Group entities by type
        standards = [e for e in entities if e.entity_type == "Standard"]
        test_methods = [e for e in entities if e.entity_type == "TestMethod"]
        parameters = [e for e in entities if e.entity_type == "Parameter"]
        soil_types = [e for e in entities if e.entity_type == "SoilType"]
        organizations = [e for e in entities if e.entity_type == "Organization"]
        people = [e for e in entities if e.entity_type == "Person"]

        # Standards REFERENCES test methods mentioned together
        for std in standards:
            for tm in test_methods:
                if self._entities_near(std.name, tm.name, text, window=500):
                    relationships.append(
                        Relationship(source=std.name, target=tm.name, rel_type="REFERENCES")
                    )

        # Test methods MEASURE parameters
        for tm in test_methods:
            for param in parameters:
                if self._entities_near(tm.name, param.name, text, window=300):
                    relationships.append(
                        Relationship(source=tm.name, target=param.name, rel_type="MEASURES")
                    )

        # Test methods APPLY_TO classification types
        for tm in test_methods:
            for soil in soil_types:
                if self._entities_near(tm.name, soil.name, text, window=400):
                    relationships.append(
                        Relationship(source=tm.name, target=soil.name, rel_type="APPLIES_TO")
                    )

        # Organizations PUBLISHES standards
        for org in organizations:
            for std in standards:
                if self._entities_near(org.name, std.name, text, window=500):
                    relationships.append(
                        Relationship(source=org.name, target=std.name, rel_type="PUBLISHES")
                    )

        # People AFFILIATED_WITH organizations
        for person in people:
            for org in organizations:
                if self._entities_near(person.name, org.name, text, window=400):
                    relationships.append(
                        Relationship(source=person.name, target=org.name, rel_type="AFFILIATED_WITH")
                    )

        return relationships

    def _entities_near(
        self,
        entity1: str,
        entity2: str,
        text: str,
        window: int = 300,
    ) -> bool:
        """Check if two entities appear within window characters."""
        text_lower = text.lower()
        e1_lower = entity1.lower()
        e2_lower = entity2.lower()

        # Find all positions of entity1
        pos1 = [m.start() for m in re.finditer(re.escape(e1_lower), text_lower)]

        if not pos1:
            return False

        # Check if entity2 appears within window of any position
        for p in pos1:
            start = max(0, p - window)
            end = min(len(text), p + len(entity1) + window)
            if e2_lower in text_lower[start:end]:
                return True

        return False

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}

        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def extract_batch(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[ExtractionResult]:
        """
        Extract entities from multiple chunks.

        Args:
            chunks: List of chunks with 'content' and 'chunk_id' keys

        Returns:
            List of ExtractionResults
        """
        results = []

        for chunk in chunks:
            content = chunk.get("content", "")
            chunk_id = chunk.get("chunk_id") or chunk.get("id")

            result = self.extract(content, chunk_id)
            results.append(result)

        return results
