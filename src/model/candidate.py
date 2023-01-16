from dataclasses import dataclass
from typing import Mapping, Any


@dataclass
class SimilarTagCandidate:
    tag: str
    features: Mapping[str, Any]
