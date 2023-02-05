import logging
from typing import List

from src.candidate_source.candidate_source import SimilarTagCandidateSource
from src.embeddings.txtai import TxtaiEmbeddings
from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest

logger = logging.getLogger("uvicorn")


class PopularTagsANNSimilarTagCandidateSource(SimilarTagCandidateSource):

    def __init__(self, embeddings: TxtaiEmbeddings):
        self.embeddings = embeddings

    async def get_candidates(
        self,
        request: SimilarTagRequest
    ) -> List[SimilarTagCandidate]:
        return [
            SimilarTagCandidate(tag_metadata.tag, {
                "poptags_score": score,
                "poptags_count": tag_metadata.count
            })
            for tag_metadata, score in self.embeddings.get_neighbors(request.tags, request.limit)
        ]
