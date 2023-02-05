from typing import List

from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest
from src.model.response import SimilarTagResponse, ScoredTag


class SimilarTagDomainResponseMarshaller:
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def marshal(
        self,
        request: SimilarTagRequest,
        candidates: List[SimilarTagCandidate]
    ) -> SimilarTagResponse:
        return SimilarTagResponse(
            scored_tags=[
                ScoredTag(candidate.tag, candidate.features.get(self.feature_name, 0))
                for i, candidate in enumerate(candidates)
            ]
        )
