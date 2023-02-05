"""SimilarTagRecommendationPipeline

Composition layer that orchestrates the different components of the pipeline.

"""
import asyncio
from typing import List

from src.candidate_source.candidate_source import SimilarTagCandidateSource
from src.candidate_source.poptags_ann_candidate_source import PopularTagsANNSimilarTagCandidateSource
from src.candidate_source.surprise_candidate_source import KNNSimilarTagCandidateSource
from src.marshaller.domain_response import SimilarTagDomainResponseMarshaller
from src.marshaller.request import SimilarTagRequestUnmarshaller
from src.marshaller.transport_response import SimilarTagTransportResponseMarshaller
from src.model.request import SimilarTagRequest
from src.model.response import SimilarTagResponse

from src.scorer.cosinesim_scorer import CosineSimScorer
from src.selector.deduping_selector import DedupingSelector
from src.selector.topk_selector import TopKSelector


class SimilarTagRecommendationPipeline:
    def __init__(
        self,
        unmarshaller: SimilarTagRequestUnmarshaller,
        knn_candidate_source: KNNSimilarTagCandidateSource,
        poptags_candidate_source: PopularTagsANNSimilarTagCandidateSource,
        cosinesim_scorer: CosineSimScorer,
        deduping_selector: DedupingSelector,
        topk_selector: TopKSelector,
        domain_marshaller: SimilarTagDomainResponseMarshaller,
        transport_marshaller: SimilarTagTransportResponseMarshaller
    ):
        self.unmarshaller = unmarshaller
        self.candidate_sources: List[SimilarTagCandidateSource] = [
            poptags_candidate_source,
            knn_candidate_source,
        ]
        self.scorers = [
            cosinesim_scorer,
        ]
        self.selectors = [
            deduping_selector,
            topk_selector,
        ]
        self.domain_marshaller = domain_marshaller
        self.transport_marshaller = transport_marshaller

    async def execute(self, params: dict):
        request: SimilarTagRequest = self.unmarshaller.unmarshal(params)

        # TODO: request hydration

        candidate_source_results = await asyncio.gather(*[
            candidate_source.get_candidates(request)
            for candidate_source in self.candidate_sources
        ])

        flattened_candidates = []
        for candidates in candidate_source_results:
            flattened_candidates += candidates

        scored_candidates = flattened_candidates
        for scorer in self.scorers:
            scored_candidates = scorer.score(request, scored_candidates)

        selected_candidates = scored_candidates
        for selector in self.selectors:
            selected_candidates = selector.select(request, selected_candidates)

        response: SimilarTagResponse = self.domain_marshaller.marshal(request, selected_candidates)

        return self.transport_marshaller.marshal(response)
