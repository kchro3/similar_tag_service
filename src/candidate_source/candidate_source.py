"""SimilarTagCandidateSource

Each candidate source should asynchronously fetch candidates given some request.

"""
import abc
from typing import List

from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest


class SimilarTagCandidateSource(abc.ABC):
    @abc.abstractmethod
    async def get_candidates(self, request: SimilarTagRequest) -> List[SimilarTagCandidate]:
        pass
