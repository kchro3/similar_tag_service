from typing import List

from src.embeddings.txtai import TxtaiEmbeddings
from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest
from src.scorer.scorer import Scorer


class CosineSimScorer(Scorer):
    """
    Hydrates scores if they don't exist
    """

    def __init__(self, embeddings: TxtaiEmbeddings, feature_name: str):
        self.embeddings = embeddings
        self.feature_name = feature_name

    def score(
        self,
        request: SimilarTagRequest,
        candidates: List[SimilarTagCandidate]
    ) -> List[SimilarTagCandidate]:
        candidates_to_hydrate = [
            candidate
            for candidate in candidates
            if self.feature_name not in candidate.features
        ]

        if len(candidates_to_hydrate) > 0:
            scores = self.embeddings.score_candidates(request.tags, map(lambda x: x.tag, candidates_to_hydrate))
            for candidate, score in zip(candidates_to_hydrate, scores):
                candidate.features[self.feature_name] = score

        return candidates
