from dataclasses import dataclass
from typing import List


@dataclass
class SimilarTagRequest:
    tags: List[str]
    limit: int
