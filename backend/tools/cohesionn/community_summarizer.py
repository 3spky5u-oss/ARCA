"""
Community Summarizer - Generate summaries for detected communities

Creates LLM summaries for communities to enable global search.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import os

from .community_detection import Community

logger = logging.getLogger(__name__)


@dataclass
class CommunitySummary:
    """Summary for a community."""

    community_id: str
    level: str
    summary: str
    themes: List[str]  # Key themes extracted from summary
    key_entities: List[Dict[str, Any]]  # Important entities with types
    node_count: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunitySummarizer:
    """
    Generate summaries for graph communities.

    Process:
    1. Extract key entities and relationships from community subgraph
    2. Gather source text excerpts from linked chunks
    3. Generate 3-5 paragraph summary with themes

    Stores summaries in Qdrant for semantic search.
    """

    # Summary prompt template
    SUMMARY_PROMPT = """Analyze the following technical content from a knowledge community.

Key Entities in this community:
{entities}

Sample content from documents:
{content}

Generate a comprehensive summary (3-5 paragraphs) that:
1. Identifies the main theme or topic of this community
2. Explains key concepts and their relationships
3. Highlights important standards, test methods, or parameters
4. Notes practical applications or considerations

Also extract 3-5 key themes as short phrases.

Format your response as:
SUMMARY:
[Your summary here]

THEMES:
- [Theme 1]
- [Theme 2]
- [Theme 3]
"""

    def __init__(
        self,
        model: str = None,
        max_content_length: int = 8000,
    ):
        """
        Args:
            model: Model name for summarization
            max_content_length: Max characters of source content to include
        """
        self.model = model or os.environ.get("COMMUNITY_SUMMARY_MODEL", "qwen2.5:1.5b")
        self.max_content_length = max_content_length
        self._client = None

    @property
    def client(self):
        """Lazy-load LLM client."""
        if self._client is None:
            from utils.llm import get_llm_client
            self._client = get_llm_client("chat")
        return self._client

    def summarize_community(
        self,
        community: Community,
    ) -> CommunitySummary:
        """
        Generate summary for a community.

        Args:
            community: Community object with node and entity IDs

        Returns:
            CommunitySummary with summary text and themes
        """
        logger.debug(f"Summarizing community {community.community_id} ({community.node_count} nodes)")

        try:
            # Get content from chunks
            content = self._get_community_content(community)

            # Format entities
            entities_str = self._format_entities(community.entity_ids)

            # Generate summary
            summary_text, themes = self._generate_summary(entities_str, content)

            # Generate embedding for the summary
            embedding = self._generate_embedding(summary_text)

            # Build key entities list
            key_entities = [
                {"name": e, "type": "entity"}
                for e in community.entity_ids[:10]
            ]

            return CommunitySummary(
                community_id=community.community_id,
                level=community.level,
                summary=summary_text,
                themes=themes,
                key_entities=key_entities,
                node_count=community.node_count,
                embedding=embedding,
                metadata={
                    "entity_count": community.entity_count,
                    "chunk_ids": community.node_ids[:20],  # Store sample of chunk IDs
                },
            )

        except Exception as e:
            logger.error(f"Failed to summarize community {community.community_id}: {e}")
            return CommunitySummary(
                community_id=community.community_id,
                level=community.level,
                summary=f"Community with {community.node_count} chunks and {community.entity_count} entities.",
                themes=["general"],
                key_entities=[],
                node_count=community.node_count,
            )

    def _get_community_content(self, community: Community) -> str:
        """Get sample content from community chunks."""
        try:
            from .vectorstore import get_knowledge_base
            from qdrant_client.models import Filter, FieldCondition, MatchAny

            kb = get_knowledge_base()
            client = kb.client

            # Sample chunk IDs (limit to avoid huge prompts)
            sample_ids = community.node_ids[:20]

            if not sample_ids:
                return "No content available."

            # Query Qdrant for chunk content
            # Convert hex chunk_ids to integer point IDs
            point_ids = []
            for chunk_id in sample_ids:
                try:
                    point_ids.append(int(chunk_id, 16))
                except ValueError:
                    pass

            if not point_ids:
                return "No content available."

            response = client.retrieve(
                collection_name="cohesionn",
                ids=point_ids,
                with_payload=["content"],
            )

            contents = []
            total_length = 0

            for point in response:
                content = point.payload.get("content", "")
                if content:
                    # Limit individual chunk length
                    chunk_excerpt = content[:500] + "..." if len(content) > 500 else content
                    contents.append(chunk_excerpt)
                    total_length += len(chunk_excerpt)

                    if total_length >= self.max_content_length:
                        break

            return "\n\n---\n\n".join(contents) if contents else "No content available."

        except Exception as e:
            logger.warning(f"Failed to get community content: {e}")
            return "Content unavailable."

    def _format_entities(self, entity_ids: List[str]) -> str:
        """Format entities for prompt."""
        if not entity_ids:
            return "No specific entities identified."

        # Limit to top 20 entities
        entities = entity_ids[:20]
        return ", ".join(entities)

    def _generate_summary(self, entities_str: str, content: str) -> tuple:
        """Generate summary using LLM."""
        prompt = self.SUMMARY_PROMPT.format(
            entities=entities_str,
            content=content,
        )

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical documentation specialist. Generate clear, accurate summaries of engineering content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0.3,
                    "max_tokens": 1000,
                },
            )

            response_text = response["message"]["content"].strip()

            # Parse response
            summary = ""
            themes = []

            if "SUMMARY:" in response_text:
                parts = response_text.split("THEMES:")
                summary = parts[0].replace("SUMMARY:", "").strip()

                if len(parts) > 1:
                    theme_section = parts[1]
                    themes = [
                        line.strip().lstrip("- ")
                        for line in theme_section.split("\n")
                        if line.strip() and line.strip() != "-"
                    ]
            else:
                # Fallback: use entire response as summary
                summary = response_text
                themes = ["general"]

            return summary, themes[:5]

        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return f"Community summary unavailable: {e}", ["error"]

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for summary text."""
        try:
            from .embeddings import get_embedder

            embedder = get_embedder()
            return embedder.embed_document(text)

        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def summarize_batch(
        self,
        communities: List[Community],
    ) -> List[CommunitySummary]:
        """Summarize multiple communities."""
        summaries = []
        total = len(communities)

        for i, community in enumerate(communities):
            summary = self.summarize_community(community)
            summaries.append(summary)

            if total > 10 and (i + 1) % 10 == 0:
                logger.info(f"Summarized {i + 1}/{total} communities")

        return summaries
