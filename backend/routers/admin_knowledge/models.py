from typing import List, Optional

from pydantic import BaseModel


class CreateTopicRequest(BaseModel):
    """Request to create a new topic folder."""

    name: str


class SearchTestRequest(BaseModel):
    """Request for search testing."""

    query: str
    topics: Optional[List[str]] = None
    top_k: int = 5
    include_routing: bool = True
    include_raw_scores: bool = True
