import logging
from functools import lru_cache
from typing import List

import surprise

from src.candidate_source.candidate_source import SimilarTagCandidateSource
from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest

logger = logging.getLogger("uvicorn")


@lru_cache
def _load_knn(knn_path):
    logger.info(f"Loading {knn_path}...")
    _, knn = surprise.dump.load(knn_path)
    logger.info(f"Loaded {knn_path}...")
    return knn


class KNNSimilarTagCandidateSource(SimilarTagCandidateSource):
    def __init__(self, knn_path="resources/tag2tag_knnz.surprise"):
        self.knn = _load_knn(knn_path)

    def get_neighbors(self, tag, k=5):
        """Surprise represents the users and items with internal IDs (iid)."""
        try:
            iid = self.knn.trainset.to_inner_iid(tag)
            return [
                self.knn.trainset.to_raw_iid(niid)
                for niid in self.knn.get_neighbors(iid, k)
            ]
        except KeyError:
            return []
        except ValueError:
            return []

    async def get_candidates(
        self,
        request: SimilarTagRequest
    ) -> List[SimilarTagCandidate]:
        return [
            SimilarTagCandidate(tag2tag, {})
            for tag in request.tags
            for i, tag2tag in enumerate(self.get_neighbors(tag, request.limit))
        ]
