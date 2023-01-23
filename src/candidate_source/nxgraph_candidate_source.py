import logging
import pickle
from functools import lru_cache
from typing import List, Tuple

from src.candidate_source.candidate_source import SimilarTagCandidateSource
from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest

logger = logging.getLogger("uvicorn")


@lru_cache
def _load_precomputed_tags(precomputed_path) -> dict[str, List[Tuple[str, int]]]:
    logger.info(f"Loading {precomputed_path}...")
    with open(precomputed_path, "rb") as f:
        precomputed = pickle.load(f)
    logger.info(f"Loaded {precomputed_path}...")
    return precomputed


class Post2TagGraphSimilarTagCandidateSource(SimilarTagCandidateSource):
    def __init__(self, precomputed_path="resources/precomputed_tag2tags.pkl"):
        self.precomputed = _load_precomputed_tags(precomputed_path)

    async def get_candidates(
        self,
        request: SimilarTagRequest
    ) -> List[SimilarTagCandidate]:
        return [
            SimilarTagCandidate(tag2tag, {
                "post2tag_score": count
            })
            for tag in request.tags
            for i, (tag2tag, count) in enumerate(self.precomputed.get(tag, []))
        ]
