"""
Unit tests for Cohesionn topic router.

Tests:
- Keyword-based routing
- Semantic routing
- Hybrid scoring
- Topic embeddings
"""

import pytest
from tools.cohesionn.reranker import TopicRouter, get_router


class TestTopicRouterBasics:
    """Basic functionality tests for TopicRouter."""

    def test_route_returns_topics(self):
        """Route should return list of topics."""
        router = TopicRouter()
        topics = router.route("load capacity analysis")

        assert isinstance(topics, list)
        assert len(topics) > 0
        assert all(isinstance(t, str) for t in topics)

    def test_route_returns_all_topics(self):
        """Route should return all configured topics."""
        router = TopicRouter()
        topics = router.route("general query")

        # Should include all configured topics
        configured_topics = list(router.TOPIC_KEYWORDS.keys())
        for t in configured_topics:
            assert t in topics

    def test_get_primary_topic(self):
        """get_primary_topic should return single topic."""
        router = TopicRouter()
        topic = router.get_primary_topic("technical analysis report")

        assert isinstance(topic, str)
        assert topic in router.TOPIC_KEYWORDS


class TestKeywordRouting:
    """Tests for keyword-based routing."""

    def test_keyword_routing_matches_configured_topics(self):
        """Queries with topic keywords should route to matching topic."""
        router = TopicRouter()

        # For each configured topic, build a query from its keywords
        for topic, keywords in router.TOPIC_KEYWORDS.items():
            if len(keywords) >= 2:
                # Build query from first two keywords
                query = f"{keywords[0]} {keywords[1]}"
                topics = router.route(query, use_semantic=False)
                assert topics[0] == topic, (
                    f"Query '{query}' should route to '{topic}' first, got {topics[0]}"
                )

    def test_query_with_no_keywords_returns_all_topics(self):
        """Query with no matching keywords should return all topics."""
        router = TopicRouter()

        topics = router.route("xyzzy foobar nonexistent", use_semantic=False)
        # Should return all topics (no keyword matches, so fallback ordering)
        assert len(topics) == len(router.TOPIC_KEYWORDS)


class TestHybridRouting:
    """Tests for hybrid keyword + semantic routing."""

    def test_hybrid_uses_both_scores(self, real_embedder):
        """Hybrid routing should combine keyword and semantic scores."""
        router = TopicRouter()

        # Pick first topic and its keywords for a deterministic test
        first_topic = list(router.TOPIC_KEYWORDS.keys())[0]
        keyword = router.TOPIC_KEYWORDS[first_topic][0]

        topics_hybrid = router.route(keyword, use_semantic=True)
        topics_keyword = router.route(keyword, use_semantic=False)

        # Both should return the matching topic first (strong keyword match)
        assert topics_hybrid[0] == first_topic
        assert topics_keyword[0] == first_topic

    def test_semantic_routing_handles_synonyms(self, real_embedder):
        """Semantic routing should handle query variations."""
        router = TopicRouter()

        # Different phrasings of similar concept
        queries = [
            "technical document reference",
            "standard report analysis",
            "technical specification guide",
        ]

        for query in queries:
            topics = router.route(query, use_semantic=True)
            # Should return topics without error
            assert len(topics) > 0, (
                f"Query '{query}' should return at least one topic"
            )


class TestTopicEmbeddings:
    """Tests for topic embedding pre-computation."""

    def test_warm_embeddings(self, real_embedder):
        """warm_embeddings should pre-compute topic embeddings."""
        router = TopicRouter()
        router.warm_embeddings()

        # Should have embeddings for all configured topics
        embeddings = router._get_topic_embeddings()
        for topic in router.TOPIC_DESCRIPTIONS:
            assert topic in embeddings

    def test_topic_embeddings_are_vectors(self, real_embedder):
        """Topic embeddings should be valid vectors."""
        router = TopicRouter()
        embeddings = router._get_topic_embeddings()

        for topic, emb in embeddings.items():
            assert isinstance(emb, list)
            assert len(emb) > 0
            assert all(isinstance(v, float) for v in emb)


class TestTopicDescriptions:
    """Tests for topic description content."""

    def test_all_topics_have_descriptions(self):
        """All topics should have descriptions."""
        router = TopicRouter()

        for topic in router.TOPIC_KEYWORDS.keys():
            assert topic in router.TOPIC_DESCRIPTIONS, (
                f"Topic '{topic}' missing description"
            )
            assert len(router.TOPIC_DESCRIPTIONS[topic]) > 50, (
                f"Topic '{topic}' description too short"
            )

    def test_all_topics_have_keywords(self):
        """All topics should have keyword lists."""
        router = TopicRouter()

        for topic in router.TOPIC_DESCRIPTIONS.keys():
            assert topic in router.TOPIC_KEYWORDS, (
                f"Topic '{topic}' missing keywords"
            )
            assert len(router.TOPIC_KEYWORDS[topic]) >= 10, (
                f"Topic '{topic}' has too few keywords"
            )


class TestSingletonRouter:
    """Tests for singleton router instance."""

    def test_get_router_returns_singleton(self):
        """get_router should return same instance."""
        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

    def test_singleton_router_works(self):
        """Singleton router should function correctly."""
        router = get_router()
        topics = router.route("technical analysis")
        assert len(topics) > 0


class TestEdgeCases:
    """Edge case tests for topic router."""

    def test_empty_query(self):
        """Empty query should return all topics."""
        router = TopicRouter()
        topics = router.route("")
        assert len(topics) == len(router.TOPIC_KEYWORDS)

    def test_numeric_query(self):
        """Numeric query should be handled."""
        router = TopicRouter()
        topics = router.route("V60 = 35")
        # Should not raise
        assert len(topics) > 0

    def test_special_characters(self):
        """Query with special characters should be handled."""
        router = TopicRouter()
        topics = router.route("φ = 30° (friction angle)")
        assert len(topics) > 0

    def test_mixed_topics_query(self):
        """Query mentioning multiple topics should return mixed ranking."""
        router = TopicRouter()

        # Build query from keywords of different topics
        all_keywords = []
        for keywords in router.TOPIC_KEYWORDS.values():
            if keywords:
                all_keywords.append(keywords[0])
        query = " ".join(all_keywords[:3])
        topics = router.route(query)

        # All topics should be represented
        assert len(topics) == len(router.TOPIC_KEYWORDS)
