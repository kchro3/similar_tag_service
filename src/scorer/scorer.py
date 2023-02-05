import abc
from typing import List, Tuple

from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest


class Scorer(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        request: SimilarTagRequest,
        candidates: List[SimilarTagCandidate]
    ) -> List[SimilarTagCandidate]:
        """Hydrate candidates to keep"""
        pass
