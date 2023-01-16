from dataclasses import dataclass
from typing import List


@dataclass
class ScoredTag:
    tag: str
    score: float


@dataclass
class SimilarTagResponse:
    scored_tags: List[ScoredTag]
