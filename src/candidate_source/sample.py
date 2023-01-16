import math
from typing import List

from src.candidate_source.candidate_source import SimilarTagCandidateSource
from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest


class SampleSimilarTagCandidateSource(SimilarTagCandidateSource):
    async def get_candidates(
        self,
        request: SimilarTagRequest
    ) -> List[SimilarTagCandidate]:
        return [
            SimilarTagCandidate(tag, {
                "sample_score": (math.e ** -i)
            })
            for i, tag in enumerate(request.tags)
        ]
