"""
Query Synthesizer — generates benchmark queries from ground truth data.

Produces ~40 queries across 5 tiers:
  - factual (15): Direct project/parameter lookups
  - neighborhood (8): Neighborhood profile descriptions
  - cross_ref (8): Cross-neighborhood comparisons
  - multi_hop (6): Multi-document reasoning
  - negation (3): Negation/exclusion queries
"""
import logging
from typing import List

from .battery import BenchmarkQuery
from .ground_truth import GroundTruthCorpus

logger = logging.getLogger(__name__)


class QuerySynthesizer:
    """Generate benchmark queries from parsed ground truth."""

    def synthesize(self, corpus: GroundTruthCorpus) -> List[BenchmarkQuery]:
        """Generate all benchmark queries from ground truth corpus."""
        queries = []
        queries.extend(self._factual_queries(corpus))
        queries.extend(self._neighborhood_queries(corpus))
        queries.extend(self._cross_ref_queries(corpus))
        queries.extend(self._multi_hop_queries(corpus))
        queries.extend(self._negation_queries(corpus))

        logger.info(f"Synthesized {len(queries)} benchmark queries across {len(set(q.tier for q in queries))} tiers")
        return queries

    def _factual_queries(self, corpus: GroundTruthCorpus) -> List[BenchmarkQuery]:
        """Generate ~15 factual lookup queries."""
        queries = []

        # Project-specific queries from ground truth projects
        project_queries = [
            # Direct project lookups
            ("fact_01", "What foundation type was recommended for the City Centre Tower project?",
             ["belled piles", "bedrock", "24m"], ["BG11-005", "City Centre"], ["BG11-005"]),
            ("fact_02", "What was the frost penetration depth used in Ridgeford designs?",
             ["2.5", "frost", "penetration", "1.8"], [], []),
            ("fact_03", "What drilling subcontractor does Broadleaf Geotechnical use?",
             ["Roughneck", "drilling", "Prairie Foundations"], ["Roughneck Drilling"], []),
            ("fact_04", "What is the groundwater depth in Crystal Shores?",
             ["surface", "1.0", "water table"], ["Crystal Shores"], []),
            ("fact_05", "What contamination was found at the Historic Gas Station in Westmount?",
             ["benzene", "contamination", "migrated"], ["BG16-112"], ["BG16-112"]),
            ("fact_06", "What foundation type is used in Ironwood Flats?",
             ["driven", "steel", "piles", "H-piles", "pipe"], ["Ironwood Flats"], []),
            ("fact_07", "What was the key finding for Harvest Hills Subdivision A?",
             ["high plastic clay", "void forms", "grade beams"], ["BG12-150"], ["BG12-150"]),
            ("fact_08", "What is the depth to bedrock in City Centre?",
             ["22", "bedrock", "Paskapoo"], ["City Centre"], []),
            ("fact_09", "What cement type is required in Crystal Shores?",
             ["sulfate-resistant", "Type HS"], ["Crystal Shores"], []),
            ("fact_10", "What lab test determines clay plasticity?",
             ["Atterberg", "limits", "ASTM D4318", "plasticity"], [], []),
            ("fact_11", "What year was Broadleaf Geotechnical established?",
             ["1987"], ["Broadleaf"], []),
            ("fact_12", "What is the seismic data for Ridgeford?",
             ["low seismicity", "0.15g"], [], []),
            ("fact_13", "What was recommended for the Valleyview Custom Home?",
             ["setback", "15m", "friction piles", "crest"], ["BG18-201"], ["BG18-201"]),
            ("fact_14", "What type of fill was found in Railtown?",
             ["ash", "clinker", "brick", "uncontrolled"], ["Railtown"], []),
            ("fact_15", "What was the bearing capacity at Aspen Ridge School?",
             ["200 kPa", "till", "standard footings"], ["BG15-060"], ["BG15-060"]),
        ]

        for qid, query, keywords, entities, sources in project_queries:
            queries.append(BenchmarkQuery(
                id=qid,
                query=query,
                tier="factual",
                difficulty=1,
                expect_keywords=keywords,
                expect_entities=entities,
                ground_truth_answer=self._build_ground_truth(corpus, keywords, sources),
                source_projects=sources,
            ))

        return queries

    def _neighborhood_queries(self, corpus: GroundTruthCorpus) -> List[BenchmarkQuery]:
        """Generate ~8 neighborhood description queries."""
        queries = []

        neighborhood_queries = [
            ("nbr_01", "Describe the stratigraphy in Crystal Shores",
             "Crystal Shores", ["peat", "organic", "lacustrine", "clay", "till"]),
            ("nbr_02", "What are the geotechnical challenges in Ironwood Flats?",
             "Ironwood Flats", ["contamination", "water table", "dewatering", "liquefaction"]),
            ("nbr_03", "What soil conditions exist in Westmount?",
             "Westmount", ["glaciolacustrine", "clay", "till", "bedrock"]),
            ("nbr_04", "Describe the foundation requirements for Valleyview Heights",
             "Valleyview Heights", ["piles", "socketed", "rock", "slope", "stability"]),
            ("nbr_05", "What challenges exist for construction in Railtown?",
             "Railtown", ["contamination", "creosote", "fill", "methane"]),
            ("nbr_06", "What are the soil conditions in Aspen Ridge?",
             "Aspen Ridge", ["silty sand", "till", "frost heave"]),
            ("nbr_07", "Describe the groundwater conditions in City Centre",
             "City Centre", ["6.0", "deep excavations", "shoring"]),
            ("nbr_08", "What are the clay properties in Harvest Hills?",
             "Harvest Hills", ["swelling", "plasticity", "PI", "void forms"]),
        ]

        for qid, query, neighborhood, keywords in neighborhood_queries:
            profile = corpus.get_neighborhood(neighborhood)
            entities = [neighborhood]
            gt_answer = ""
            if profile:
                gt_answer = f"{profile.stratigraphy} {profile.challenges} {profile.foundations}"

            queries.append(BenchmarkQuery(
                id=qid,
                query=query,
                tier="neighborhood",
                difficulty=2,
                expect_keywords=keywords,
                expect_entities=entities,
                ground_truth_answer=gt_answer,
                source_projects=[p.project_num for p in corpus.get_projects_by_neighborhood(neighborhood)],
            ))

        return queries

    def _cross_ref_queries(self, corpus: GroundTruthCorpus) -> List[BenchmarkQuery]:
        """Generate ~8 cross-reference comparison queries."""
        queries = []

        cross_queries = [
            ("xref_01", "Compare the foundation types used in Ironwood Flats versus City Centre",
             ["driven", "piles", "belled", "bedrock", "H-piles"],
             ["Ironwood Flats", "City Centre"]),
            ("xref_02", "How do groundwater conditions differ between Crystal Shores and Aspen Ridge?",
             ["surface", "1.0", "deep", "7.0", "water table"],
             ["Crystal Shores", "Aspen Ridge"]),
            ("xref_03", "Compare the contamination issues between Railtown and Ironwood Flats",
             ["creosote", "hydrocarbons", "metals", "contamination"],
             ["Railtown", "Ironwood Flats"]),
            ("xref_04", "What are the differences in clay properties between Westmount and Harvest Hills?",
             ["plasticity", "clay", "glaciolacustrine", "swelling"],
             ["Westmount", "Harvest Hills"]),
            ("xref_05", "Compare slope stability concerns between Valleyview Heights and other neighborhoods",
             ["slope", "landslide", "colluvium", "stability"],
             ["Valleyview Heights"]),
            ("xref_06", "How do foundation designs differ between residential and commercial developments?",
             ["footings", "piles", "strip", "belled", "raft"],
             []),
            ("xref_07", "Compare the bedrock depth and type across different neighborhoods",
             ["bedrock", "Paskapoo", "mudstone", "sandstone"],
             []),
            ("xref_08", "What environmental assessment types were conducted across the projects?",
             ["Phase I", "Phase II", "ESA", "contamination", "remediation"],
             []),
        ]

        for qid, query, keywords, entities in cross_queries:
            queries.append(BenchmarkQuery(
                id=qid,
                query=query,
                tier="cross_ref",
                difficulty=2,
                expect_keywords=keywords,
                expect_entities=entities,
                ground_truth_answer="",
                source_projects=[],
            ))

        return queries

    def _multi_hop_queries(self, corpus: GroundTruthCorpus) -> List[BenchmarkQuery]:
        """Generate ~6 multi-hop reasoning queries."""
        queries = []

        hop_queries = [
            ("hop_01", "Which projects involved slope stability assessments and what were the outcomes?",
             ["BG09-044", "BG22-055", "BG25-010", "slope", "landslide", "retaining", "shear key"],
             ["BG09-044", "BG22-055", "BG25-010"]),
            ("hop_02", "Trace the history of the Ironwood Industrial Lot from Phase I ESA to Phase II completion",
             ["BG23-001", "BG23-002", "metals", "welding", "guidelines"],
             ["BG23-001", "BG23-002"]),
            ("hop_03", "What is the relationship between soil type and foundation recommendations across all neighborhoods?",
             ["clay", "till", "piles", "footings", "bearing"],
             []),
            ("hop_04", "Track the Railtown redevelopment from remediation through condo proposal to investigation",
             ["BG14-012", "BG14-015", "BG20-022", "BG25-045", "contamination", "screw piles"],
             ["BG14-012", "BG14-015", "BG20-022", "BG25-045"]),
            ("hop_05", "How have foundation design practices evolved across the project history?",
             ["piles", "footings", "micropiles", "screw", "driven"],
             []),
            ("hop_06", "Which projects required special cement types and why?",
             ["sulfate", "Type HS", "CSA A23.1", "Crystal Shores"],
             []),
        ]

        for qid, query, keywords, sources in hop_queries:
            queries.append(BenchmarkQuery(
                id=qid,
                query=query,
                tier="multi_hop",
                difficulty=3,
                expect_keywords=keywords,
                expect_entities=[],
                ground_truth_answer="",
                source_projects=sources,
            ))

        return queries

    def _negation_queries(self, corpus: GroundTruthCorpus) -> List[BenchmarkQuery]:
        """Generate ~3 negation queries."""
        queries = []

        neg_queries = [
            ("neg_01", "Which neighborhoods do NOT have a high water table?",
             ["Aspen Ridge", "Westmount", "City Centre", "Harvest Hills"],
             ["Aspen Ridge", "Westmount"]),
            ("neg_02", "Which neighborhoods have no reported contamination issues?",
             ["Westmount", "Aspen Ridge", "Valleyview", "Crystal Shores", "Harvest Hills", "City Centre"],
             []),
            ("neg_03", "Which project areas do NOT require pile foundations?",
             ["Westmount", "Aspen Ridge", "Harvest Hills"],
             []),
        ]

        for qid, query, keywords, entities in neg_queries:
            queries.append(BenchmarkQuery(
                id=qid,
                query=query,
                tier="negation",
                difficulty=3,
                expect_keywords=keywords,
                expect_entities=entities,
                ground_truth_answer="",
                source_projects=[],
            ))

        return queries

    def _build_ground_truth(self, corpus: GroundTruthCorpus, keywords: List[str], source_projects: List[str]) -> str:
        """Build a ground truth answer string from project records."""
        if not source_projects:
            return ""

        parts = []
        for proj_num in source_projects:
            for proj in corpus.projects:
                if proj.project_num == proj_num:
                    parts.append(f"{proj.project_name}: {proj.key_finding}")
                    break

        return ". ".join(parts)
