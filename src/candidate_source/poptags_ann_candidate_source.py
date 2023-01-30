import logging
import pickle
from functools import lru_cache
from typing import List
from dataclasses import dataclass

from txtai.embeddings import Embeddings

from src.candidate_source.candidate_source import SimilarTagCandidateSource
from src.model.candidate import SimilarTagCandidate
from src.model.request import SimilarTagRequest

logger = logging.getLogger("uvicorn")


@dataclass
class TagMetadata:
    tag: str
    count: int


@lru_cache
def _load_embeddings(model_path, embedding_path, embedding_metadata_path):
    logger.info(f"Loading {model_path}...")
    embedding = Embeddings({
        "path": model_path
    })

    logger.info(f"Loading {embedding_path}...")
    embedding.load(embedding_path)

    logger.info(f"Loading {embedding_metadata_path}...")
    with open(embedding_metadata_path, 'rb') as f:
        embedding_metadata = pickle.load(f)

    logger.info(f"Loaded embeddings.")
    return embedding, embedding_metadata


class PopularTagsANNSimilarTagCandidateSource(SimilarTagCandidateSource):
    MIN_SCORE = 0.5
    MAX_SCORE = 0.9

    def __init__(
        self,
        model_path="sentence-transformers/all-mpnet-base-v2",
        embedding_path="resources/popular_tags.txtai",
        metadata_path="resources/popular_tags_metadata.pkl"
    ):
        self.embeddings, metadata = _load_embeddings(model_path, embedding_path, metadata_path)
        self.embedding_metadata = {}
        for docid, tag_data in metadata.items():
            self.embedding_metadata[docid] = TagMetadata(tag_data['tag'], tag_data['count'])

    def get_neighbors(self, tag, k=10):
        tags_and_scores = []
        for doc, score in self.embeddings.search(tag, k):
            if self.MIN_SCORE < score < self.MAX_SCORE:
                tags_and_scores.append((self.embedding_metadata[doc], score))
        return tags_and_scores

    async def get_candidates(
        self,
        request: SimilarTagRequest
    ) -> List[SimilarTagCandidate]:
        return [
            SimilarTagCandidate(tag_metadata.tag, {
                "poptags_score": score,
                "poptags_count": tag_metadata.count
            })
            for tag in request.tags
            for tag_metadata, score in self.get_neighbors(tag, request.limit)
        ]
