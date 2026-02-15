"""
Query Classifier - Selects the appropriate handler for a query.

Iterates handlers by priority (lowest first), returns first match.
"""

import logging
from typing import List, Optional

from .base import QueryHandler, HandlerContext

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classifies queries and routes to appropriate handlers.

    Usage:
        classifier = QueryClassifier()
        classifier.register(GeologyHandler())
        classifier.register(TechnicalHandler())
        classifier.register(DefaultHandler())

        handler = classifier.classify(ctx)
        await handler.pre_process(ctx)
    """

    def __init__(self):
        self._handlers: List[QueryHandler] = []
        self._sorted = False

    def register(self, handler: QueryHandler) -> None:
        """Register a handler."""
        self._handlers.append(handler)
        self._sorted = False
        logger.debug(f"Registered handler: {handler.name} (priority {handler.priority})")

    def _ensure_sorted(self) -> None:
        """Ensure handlers are sorted by priority."""
        if not self._sorted:
            self._handlers.sort(key=lambda h: h.priority)
            self._sorted = True

    def classify(self, ctx: HandlerContext) -> QueryHandler:
        """
        Find the appropriate handler for a query.

        Args:
            ctx: Handler context with message and router decision

        Returns:
            The handler that should process this query
        """
        self._ensure_sorted()

        for handler in self._handlers:
            if handler.should_handle(ctx):
                ctx.handler_name = handler.name
                logger.info(f"Query classified as: {handler.name}")
                return handler

        # Should never happen if DefaultHandler is registered
        logger.warning("No handler matched - this shouldn't happen")
        return self._handlers[-1] if self._handlers else None

    def get_handlers(self) -> List[QueryHandler]:
        """Get all registered handlers (sorted by priority)."""
        self._ensure_sorted()
        return self._handlers.copy()


# Global classifier instance with all handlers registered
_classifier: Optional[QueryClassifier] = None


def get_classifier() -> QueryClassifier:
    """Get or create the global classifier with all handlers registered."""
    global _classifier

    if _classifier is None:
        from .default import DefaultHandler
        from .technical import TechnicalHandler
        from .calculate import CalculateHandler
        from .think import ThinkHandler

        _classifier = QueryClassifier()

        # Register in priority order (but classifier sorts anyway)
        # GeologyHandler requires mapperr (domain pack with mapping support) — skip if unavailable
        try:
            from .geology import GeologyHandler
            _classifier.register(GeologyHandler())
        except ImportError:
            pass  # Geology handler unavailable (no mapperr domain tool)

        _classifier.register(CalculateHandler())
        _classifier.register(ThinkHandler())
        _classifier.register(TechnicalHandler())
        _classifier.register(DefaultHandler())

        logger.info(f"QueryClassifier initialized with {len(_classifier._handlers)} handlers")

    return _classifier
