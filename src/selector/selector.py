import abc
from typing import List, Tuple

from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest


class Selector(abc.ABC):
    @abc.abstractmethod
    def select(
        self,
        request: SimilarTagRequest,
        candidates: List[SimilarTagCandidate]
    ) -> List[SimilarTagCandidate]:
        """Only return candidates to keep"""
        pass
