from dataclasses import dataclass
from typing import Any


@dataclass
class SimilarTagCandidate:
    tag: str
    features: dict[str, Any]
