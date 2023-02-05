import logging
import pickle
from dataclasses import dataclass
from typing import List

import numpy as np
from txtai.embeddings import Embeddings


@dataclass
class TagMetadata:
    tag: str
    count: int


logger = logging.getLogger("uvicorn")


class TxtaiEmbeddings:
    """Wrapper class for handling txtai Embeddings"""
    MIN_SCORE = 0.5
    MAX_SCORE = 0.9

    def __init__(self, model_path, embedding_path, metadata_path):
        logger.info(f"Loading {model_path}...")
        self.embeddings = Embeddings({
            "path": model_path
        })

        logger.info(f"Loading {embedding_path}...")
        self.embeddings.load(embedding_path)
        logger.info(f"Loaded embeddings.")

        logger.info(f"Loading {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        self.embedding_metadata = {}
        for docid, tag_data in metadata.items():
            self.embedding_metadata[docid] = TagMetadata(tag_data['tag'], tag_data['count'])

    def get_neighbors(self, tags, k=10):
        tags_and_scores = []
        for hits in self.embeddings.batchsearch(tags, k):
            for doc, score in hits:
                if self.MIN_SCORE < score < self.MAX_SCORE:
                    tags_and_scores.append((self.embedding_metadata[doc], score))
        return tags_and_scores

    def score_candidates(self, queries: List[str], data: List[str]):
        # Convert queries to embedding vectors
        queries = self.embeddings.batchtransform((None, query, None) for query in queries)
        data = self.embeddings.batchtransform((None, row, None) for row in data)

        # Dot product on normalized vectors is equal to cosine similarity
        scores = np.dot(queries, data.T).tolist()
        # Take the mean score over the queries
        mean_scores = np.mean(scores, axis=0)
        return mean_scores
