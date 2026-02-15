"""
Cohesionn Retriever - Main search pipeline with citations

Combines routing, retrieval, and reranking for technical Q&A.
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .vectorstore import get_knowledge_base, KnowledgeBase
from .reranker import get_reranker, get_router, get_diversity_reranker, BGEReranker, TopicRouter, DiversityReranker
from .sparse_retrieval import get_bm25_manager, reciprocal_rank_fusion, BM25Manager
from .query_expansion import get_query_expander, QueryExpander
from .hyde import get_hyde_generator, HyDEGenerator
from .crag import get_crag_evaluator, CRAGEvaluator
from .raptor import RaptorRetrieverMixin
from .graph_retriever import get_graph_retriever, GraphRetriever
from .global_retriever import get_global_retriever, GlobalRetriever

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation for retrieved content"""

    source: str
    title: Optional[str]
    page: Optional[int]
    section: Optional[str]
    topic: str

    def format_inline(self) -> str:
        """Short inline citation: [Title, p.X]"""
        parts = []
        if self.title:
            # Shorten long titles
            title = self.title[:50] + "..." if len(self.title) > 50 else self.title
            parts.append(title)
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(self.section[:30])

        return f"[{', '.join(parts)}]" if parts else "[Unknown]"

    def format_reference(self, index: int) -> str:
        """Full reference: [1] Title. Section. Page X."""
        parts = [f"[{index}]"]
        if self.title:
            parts.append(self.title)
        if self.section:
            parts.append(f"Section: {self.section}")
        if self.page:
            parts.append(f"Page {self.page}")

        return " ".join(parts)


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline"""

    query: str
    chunks: List[Dict[str, Any]]
    citations: List[Citation]
    topics_searched: List[str]
    confidence: str = "unknown"   # "high" / "medium" / "low"
    max_score: float = 0.0

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0

    def get_context(self, max_chunks: int = 5) -> str:
        """
        Get formatted context for LLM consumption.

        Each chunk is prefixed with its citation for easy reference.
        """
        if not self.chunks:
            return "No relevant information found in knowledge base."

        parts = []
        for i, chunk in enumerate(self.chunks[:max_chunks]):
            cite = self.citations[i] if i < len(self.citations) else None
            cite_str = cite.format_inline() if cite else f"[Source {i+1}]"

            parts.append(f"--- {cite_str} ---\n{chunk['content']}")

        return "\n\n".join(parts)

    def get_references(self) -> str:
        """Get formatted reference list"""
        if not self.citations:
            return ""

        refs = ["References:"]
        seen = set()

        for i, citation in enumerate(self.citations, 1):
            # Dedupe by source+page
            key = (citation.source, citation.page)
            if key in seen:
                continue
            seen.add(key)

            refs.append(citation.format_reference(i))

        return "\n".join(refs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return {
            "query": self.query,
            "topics_searched": self.topics_searched,
            "num_results": len(self.chunks),
            "chunks": [
                {
                    "content": c["content"][:500] + "..." if len(c["content"]) > 500 else c["content"],
                    "score": self._normalize_score(c.get("rerank_score", c.get("score", 0))),
                    "source": self._clean_source_title(c["metadata"].get("source", "")),
                    "page": c["metadata"].get("page"),
                    "section": c["metadata"].get("section"),
                    "topic": c.get("topic", "unknown"),
                }
                for c in self.chunks
            ],
            "confidence": self.confidence,
            "max_score": round(self.max_score, 4),
            "context": self.get_context(),
            "references": self.get_references(),
        }

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """
        Normalize retrieval scores to a more intuitive 0-1 range.

        Raw cosine similarity from dense embeddings is typically 0.1-0.5 for good matches.
        Cross-encoder scores are often 0-1 but clustered low.

        Recalibrated for stricter/more honest scoring:
        - 0.45+ raw → 0.90+ (excellent match)
        - 0.35  raw → 0.80  (very good match)
        - 0.25  raw → 0.65  (good match)
        - 0.15  raw → 0.40  (moderate match)
        - 0.10  raw → 0.25  (weak match)
        - <0.10 raw → <0.25 (poor match)
        """
        if raw_score >= 0.45:
            return min(0.98, 0.90 + (raw_score - 0.45) * 0.4)  # 0.45-0.65 → 0.90-0.98
        elif raw_score >= 0.35:
            return 0.80 + (raw_score - 0.35) * 1.0  # 0.35-0.45 → 0.80-0.90
        elif raw_score >= 0.25:
            return 0.65 + (raw_score - 0.25) * 1.5  # 0.25-0.35 → 0.65-0.80
        elif raw_score >= 0.15:
            return 0.40 + (raw_score - 0.15) * 2.5  # 0.15-0.25 → 0.40-0.65
        elif raw_score >= 0.10:
            return 0.25 + (raw_score - 0.10) * 3.0  # 0.10-0.15 → 0.25-0.40
        else:
            return raw_score * 2.5  # 0-0.10 → 0-0.25

    @staticmethod
    def _clean_source_title(source: str) -> str:
        """Extract clean book/document title from file path."""
        from pathlib import Path as PathLib

        if not source:
            return "Unknown Source"

        # Get filename without path and extension
        name = PathLib(source).stem

        # If still looks like a path, try .name
        if not name or "\\" in name or "/" in name:
            name = PathLib(source).name
            name = PathLib(name).stem  # Remove extension

        # Remove common prefixes
        name = re.sub(r"^\d{10,}-", "", name)  # ISBN prefixes
        name = re.sub(r"^\[\d+\]", "", name)  # [1234] prefixes

        # Remove year suffixes like (2014) or _2014
        name = re.sub(r"[\(_]\d{4}[\)]?$", "", name)

        # Clean separators
        name = name.replace("_", " ")
        name = name.replace("-", " ")
        name = re.sub(r"\s+", " ", name).strip()

        # Title case if all lowercase
        if name == name.lower():
            name = name.title()

        # Truncate long titles
        if len(name) > 60:
            name = name[:57] + "..."

        return name if name else "Unknown Source"


class CohesionnRetriever(RaptorRetrieverMixin):
    """
    Main retrieval pipeline for Cohesionn.

    Flow:
    1. Expand query with engineering synonyms (Phase 1)
    2. Generate HyDE hypothetical document (Phase 2)
    3. Route query to relevant topics
    4. Retrieve dense (semantic) candidates from Qdrant
    5. Retrieve sparse (BM25) candidates for exact term matching (Phase 1)
    6. Retrieve RAPTOR hierarchical summaries (Phase 3a) - for broad queries
    7. Fuse results with Reciprocal Rank Fusion (Phase 1)
    8. Rerank combined candidates with cross-encoder
    9. Apply MMR diversity reranking (balances relevance with source diversity)
    10. Apply quality threshold (drop low-confidence results)
    11. CRAG web fallback on low confidence (Phase 2)
    12. Format with citations
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase = None,
        reranker: BGEReranker = None,
        diversity_reranker: DiversityReranker = None,
        router: TopicRouter = None,
        bm25_manager: BM25Manager = None,
        query_expander: QueryExpander = None,
        hyde_generator: HyDEGenerator = None,
        crag_evaluator: CRAGEvaluator = None,
        db_path: Path = None,
    ):
        self.kb = knowledge_base or get_knowledge_base(db_path)
        self.reranker = reranker or get_reranker()
        self.diversity_reranker = diversity_reranker  # Lazy-loaded with config
        self.router = router or get_router()
        self.bm25_manager = bm25_manager  # Lazy-loaded with config
        self.query_expander = query_expander  # Lazy-loaded with config
        self.hyde_generator = hyde_generator  # Lazy-loaded with config
        self.crag_evaluator = crag_evaluator  # Lazy-loaded with config

    def retrieve(
        self,
        query: str,
        topics: Optional[List[str]] = None,
        top_k: int = None,
        initial_k: int = None,
        rerank: bool = True,
        apply_diversity: bool = None,
        use_hybrid: bool = None,
        use_expansion: bool = None,
        use_hyde: bool = None,
        use_crag: bool = None,
        use_raptor: bool = None,
        use_graph: bool = None,
        use_global: bool = None,
        profile: str = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            topics: Specific topics to search (auto-routed if None)
            top_k: Number of final results (default from config)
            initial_k: Candidates to retrieve before reranking (default from config)
            rerank: Whether to apply cross-encoder reranking
            apply_diversity: Whether to apply MMR diversity (default from config)
            use_hybrid: Whether to use BM25+dense hybrid (default from config)
            use_expansion: Whether to expand query with synonyms (default from config)
            use_hyde: Whether to use HyDE hypothetical embedding (default from config)
            use_crag: Whether to use CRAG web fallback (default from config)
            use_raptor: Whether to use RAPTOR hierarchical retrieval (default from config)
            use_graph: Whether to use GraphRAG retrieval (default from config)
            use_global: Whether to use global/community search (default from config)

        Returns:
            RetrievalResult with chunks, citations, and formatted context
        """
        # Import config for runtime settings
        from config import runtime_config

        # Resolve toggles through profile hierarchy
        from profile_loader import get_profile_manager
        pm = get_profile_manager()
        toggles = pm.resolve_for_query(
            profile_override=profile,
            bm25_enabled=use_hybrid,
            query_expansion_enabled=use_expansion,
            hyde_enabled=use_hyde,
            crag_enabled=use_crag,
            raptor_enabled=use_raptor,
            graph_rag_enabled=use_graph,
            global_search_enabled=use_global,
            rag_diversity_enabled=apply_diversity,
            reranker_enabled=rerank if not isinstance(rerank, bool) else None,
            domain_boost_enabled=None,
        )

        # Unpack resolved toggles into local variables
        use_hybrid = toggles.get("bm25_enabled", True)
        use_expansion = toggles.get("query_expansion_enabled", True)
        use_hyde = toggles.get("hyde_enabled", True)
        use_crag = toggles.get("crag_enabled", False)
        use_raptor = toggles.get("raptor_enabled", True)
        use_graph = toggles.get("graph_rag_enabled", True)
        use_global = toggles.get("global_search_enabled", True)
        apply_diversity = toggles.get("rag_diversity_enabled", True)

        # Cross-reference query routing: activate GraphRAG for comparison queries
        # Benchmark v2: GraphRAG earns +0.024 composite on cross-ref tier
        if profile is None and runtime_config.graph_rag_auto:
            from .query_classifier import get_query_classifier
            classifier = get_query_classifier()
            if classifier.should_use_graph_search(query):
                use_graph = True
                logger.info("Cross-reference query detected, enabling GraphRAG traversal")

        # Non-toggle defaults from config
        if top_k is None:
            top_k = runtime_config.rag_top_k
        if initial_k is None:
            initial_k = runtime_config.reranker_candidates

        min_raw_score = runtime_config.rag_min_score
        graph_weight = runtime_config.graph_rag_weight
        min_final_score = runtime_config.rag_min_final_score
        bm25_weight = runtime_config.bm25_weight

        # Step 0a: Query expansion with engineering synonyms
        # Skip for short queries (< 5 words) -- expansion adds latency without benefit
        search_query = query
        if use_expansion and len(query.split()) >= 5:
            expander = self.query_expander or get_query_expander()
            search_query = expander.expand(query)
            if search_query != query:
                logger.debug(f"Query expanded: '{query}' -> '{search_query}'")

        # Step 0b: HyDE - generate hypothetical document for better semantic matching
        # Skip for short queries (< 5 words) — HyDE LLM call adds latency without benefit
        if use_hyde and len(query.split()) >= 5:
            try:
                hyde = self.hyde_generator or get_hyde_generator()
                hyde_query = hyde.expand_query(search_query)
                if hyde_query != search_query:
                    logger.debug(f"HyDE expanded query ({len(hyde_query)} chars)")
                    search_query = hyde_query
            except Exception as e:
                logger.warning(f"HyDE failed, using original query: {e}")

        # Route to topics
        auto_routed = topics is None
        if topics is None:
            topics = self.router.route(search_query)

        # Filter dead topics from auto-routing.
        # Keep arca_core so system self-knowledge remains reachable.
        if auto_routed:
            dead_topics = {"general"}
            filtered = [t for t in topics if t not in dead_topics]
            if filtered and len(filtered) < len(topics):
                removed = [t for t in topics if t in dead_topics]
                logger.debug(f"Filtered dead topics from auto-routing: {removed}")
                topics = filtered

        logger.info(f"Searching topics {topics} for: {query[:50]}...")

        # Step 1: Dense (semantic) retrieval from Qdrant
        dense_candidates = self.kb.search(
            query=search_query,
            topics=topics,
            n_results=initial_k,
        )

        # Step 2: Sparse (BM25) retrieval for exact term matching
        sparse_candidates = []
        if use_hybrid:
            try:
                bm25 = self.bm25_manager or get_bm25_manager()
                sparse_candidates = bm25.search(
                    query=search_query,
                    topics=topics,
                    n_results=initial_k,
                )
                logger.debug(f"BM25 returned {len(sparse_candidates)} candidates")
            except Exception as e:
                logger.warning(f"BM25 search failed, using dense only: {e}")

        # Step 2.5: RAPTOR hierarchical retrieval for broad/conceptual queries
        raptor_candidates = []
        if use_raptor and self.should_use_raptor(query, runtime_config):
            try:
                raptor_candidates = self.retrieve_raptor_nodes(
                    query=search_query,
                    topics=topics,
                    n_results=initial_k // 2,  # Half the candidates from RAPTOR
                    strategy="collapsed",
                )
                if raptor_candidates:
                    logger.debug(f"RAPTOR returned {len(raptor_candidates)} hierarchical summaries")
            except Exception as e:
                logger.warning(f"RAPTOR retrieval failed: {e}")

        # Step 2.6: Graph retrieval for entity-based reasoning
        graph_candidates = []
        if use_graph:
            try:
                graph_retriever = get_graph_retriever()
                graph_candidates = graph_retriever.retrieve(
                    query=query,  # Use original query for entity extraction
                    topics=topics,
                    n_results=initial_k // 2,
                )
                if graph_candidates:
                    logger.debug(f"Graph returned {len(graph_candidates)} entity-linked chunks")
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")

        # Step 2.7: Global/Community search for broad theme queries
        global_context = None
        if use_global:
            try:
                global_retriever = get_global_retriever()
                if global_retriever.should_use(query):
                    global_results = global_retriever.retrieve(
                        query=query,
                        level=runtime_config.community_level_default,
                    )
                    if global_results:
                        global_context = global_retriever.format_context(global_results)
                        logger.debug(f"Global search: {len(global_results)} community summaries")
            except Exception as e:
                logger.warning(f"Global search failed: {e}")

        # Step 3: Reciprocal Rank Fusion (dense + sparse + graph)
        if sparse_candidates or graph_candidates:
            candidates = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                graph_results=graph_candidates if graph_candidates else None,
                dense_weight=0.5 if graph_candidates else (1.0 - bm25_weight),
                sparse_weight=0.3 if graph_candidates else bm25_weight,
                graph_weight=graph_weight if graph_candidates else 0.0,
            )
            components = [f"{len(dense_candidates)} dense"]
            if sparse_candidates:
                components.append(f"{len(sparse_candidates)} sparse")
            if graph_candidates:
                components.append(f"{len(graph_candidates)} graph")
            logger.debug(f"RRF fused {' + '.join(components)} -> {len(candidates)}")
        else:
            candidates = dense_candidates

        # Step 3.5: Merge RAPTOR results if available
        if raptor_candidates:
            candidates = self.merge_raptor_results(
                dense_results=candidates,
                raptor_results=raptor_candidates,
                raptor_weight=0.3,  # 30% weight for hierarchical summaries
            )
            logger.debug(f"Merged {len(raptor_candidates)} RAPTOR results, total {len(candidates)}")

        if not candidates:
            logger.info("No candidates found")
            return RetrievalResult(
                query=query,
                chunks=[],
                citations=[],
                topics_searched=topics,
            )

        # Pre-filter: Skip candidates with very low raw scores before expensive reranking
        filtered_candidates = [c for c in candidates if c.get("score", 0) >= min_raw_score]
        if not filtered_candidates:
            # Fall back to best available if all filtered out
            filtered_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
            logger.warning(
                f"All candidates below min_raw_score={min_raw_score}, using top {len(filtered_candidates)} by score"
            )

        logger.debug(
            f"Pre-filter: {len(candidates)} → {len(filtered_candidates)} candidates (min_score={min_raw_score})"
        )

        # Step 1: Cross-encoder reranking (relevance scoring)
        if rerank and len(filtered_candidates) > 1:
            # Get more candidates for diversity pass
            rerank_top_k = max(top_k * 3, 15) if apply_diversity else top_k
            reranked = self.reranker.rerank(query, filtered_candidates, top_k=rerank_top_k)
        else:
            reranked = filtered_candidates

        # Step 1.5: Apply domain boost (counters reranker LRFD bias)
        if runtime_config.domain_boost_enabled:
            from .reranker import apply_domain_boost

            reranked = apply_domain_boost(
                reranked, query, boost_factor=runtime_config.domain_boost_factor
            )

        # Step 2: MMR diversity reranking (source diversity)
        if apply_diversity and len(reranked) > 1:
            # Lazy-load diversity reranker with current config
            diversity_reranker = self.diversity_reranker or get_diversity_reranker()
            results = diversity_reranker.rerank(reranked, top_k=top_k, score_key="rerank_score")
        else:
            results = reranked[:top_k]

        # Step 3: Quality threshold - filter out low-confidence results
        if min_final_score > 0:
            quality_filtered = []
            for r in results:
                score = r.get("rerank_score", r.get("score", 0))
                if score >= min_final_score:
                    quality_filtered.append(r)
                else:
                    logger.debug(f"Dropped result below quality threshold: score={score:.3f} < {min_final_score}")

            if quality_filtered:
                results = quality_filtered
            else:
                # Keep at least one result if all filtered
                results = results[:1] if results else []
                logger.warning(f"All results below min_final_score={min_final_score}, keeping top 1")

        # Confidence evaluation (informational — CRAG is now a manual fallback tool)
        max_score = max((r.get("rerank_score", r.get("score", 0)) for r in results), default=0.0)
        if max_score < runtime_config.rag_min_final_score:
            confidence = "low"
        elif max_score < 0.4:
            confidence = "medium"
        else:
            confidence = "high"

        # Build citations with clean titles
        citations = []
        for chunk in results:
            meta = chunk.get("metadata", {})
            raw_source = meta.get("source", "Unknown")
            # Use title if available, otherwise clean the source path
            title = meta.get("title") or RetrievalResult._clean_source_title(raw_source)
            citations.append(
                Citation(
                    source=raw_source,
                    title=title,
                    page=meta.get("page"),
                    section=meta.get("section"),
                    topic=chunk.get("topic", "unknown"),
                )
            )

        unique_sources = len(set(c.source for c in citations))
        logger.info(f"Retrieved {len(results)} chunks from {unique_sources} unique sources")

        return RetrievalResult(
            query=query,
            chunks=results,
            citations=citations,
            topics_searched=topics,
            confidence=confidence,
            max_score=max_score,
        )

    def search_topic(
        self,
        query: str,
        topic: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Search a specific topic only"""
        return self.retrieve(query, topics=[topic], top_k=top_k)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        topic_stats = self.kb.get_stats()
        return {
            "topics": topic_stats,
            "total_chunks": sum(topic_stats.values()),
        }


def search_knowledge(
    query: str,
    topics: Optional[List[str]] = None,
    top_k: int = 5,
    db_path: Path = None,
    profile: str = None,
) -> Dict[str, Any]:
    """
    Main entry point for ARCA integration.

    Args:
        query: Technical question
        topics: Optional topic filter
        top_k: Number of results
        db_path: Path to knowledge base

    Returns:
        Dict with context, references, and metadata
    """
    try:
        retriever = CohesionnRetriever(db_path=db_path)
        result = retriever.retrieve(query, topics=topics, top_k=top_k, profile=profile)

        return {
            "success": True,
            **result.to_dict(),
        }

    except Exception as e:
        logger.error(f"Knowledge search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }
