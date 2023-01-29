from typing import List

from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest
from src.model.response import SimilarTagResponse, ScoredTag


class SimilarTagDomainResponseMarshaller:
    def marshal(
        self,
        request: SimilarTagRequest,
        candidates: List[SimilarTagCandidate]
    ) -> SimilarTagResponse:
        return SimilarTagResponse(
            scored_tags=[
                ScoredTag(candidate.tag, 0)
                for i, candidate in enumerate(candidates)
            ]
        )
