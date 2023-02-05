from typing import List

from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest
from src.selector.selector import Selector


class TopKSelector(Selector):
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def select(
        self,
        request: SimilarTagRequest,
        candidates: List[SimilarTagCandidate]
    ) -> List[SimilarTagCandidate]:
        return sorted(
            candidates,
            key=lambda x: x.features[self.feature_name],
            reverse=True
        )[:request.limit]
