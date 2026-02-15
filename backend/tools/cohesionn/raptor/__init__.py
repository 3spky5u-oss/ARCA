"""
RAPTOR - Recursive Abstractive Processing for Tree-Organized Retrieval

Hierarchical summarization for answering broad/conceptual questions.

Architecture:
- Level 0: Leaf chunks (existing 1500-char chunks)
- Level 1: Cluster summaries (~10 chunks each)
- Level 2: Section summaries (~10 L1 nodes)
- Level 3: Topic summaries (optional)

Usage:
    from tools.cohesionn.raptor import RaptorTreeBuilder, RaptorRetrieverMixin

    # Build RAPTOR tree for a topic
    builder = RaptorTreeBuilder()
    stats = builder.build_tree(topic="my_topic")

    # Retrieval uses mixin integrated in main retriever
"""

from .clusterer import RaptorClusterer
from .summarizer import RaptorSummarizer
from .tree_builder import RaptorTreeBuilder, RaptorNode
from .retriever_mixin import RaptorRetrieverMixin

__all__ = [
    "RaptorClusterer",
    "RaptorSummarizer",
    "RaptorTreeBuilder",
    "RaptorNode",
    "RaptorRetrieverMixin",
]
