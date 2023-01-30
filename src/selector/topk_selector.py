from typing import List

from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest
from src.selector.selector import Selector


class TopKSelector(Selector):
    def select(
        self,
        request: SimilarTagRequest,
        candidates: List[SimilarTagCandidate]
    ) -> List[SimilarTagCandidate]:
        return candidates[:request.limit]
