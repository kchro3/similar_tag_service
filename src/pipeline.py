"""SimilarTagRecommendationPipeline

Composition layer that orchestrates the different components of the pipeline.

"""
import asyncio
from typing import List

from fastapi import Depends

from src.candidate_source.candidate_source import SimilarTagCandidateSource
from src.candidate_source.sample import SampleSimilarTagCandidateSource
from src.marshaller.domain_response import SimilarTagDomainResponseMarshaller
from src.marshaller.request import SimilarTagRequestUnmarshaller
from src.marshaller.transport_response import SimilarTagTransportResponseMarshaller
from src.model.request import SimilarTagRequest
from src.model.response import SimilarTagResponse


class SimilarTagRecommendationPipeline:
    def __init__(
        self,
        unmarshaller: SimilarTagRequestUnmarshaller = Depends(),
        sample_candidate_source: SampleSimilarTagCandidateSource = Depends(),
        domain_marshaller: SimilarTagDomainResponseMarshaller = Depends(),
        transport_marshaller: SimilarTagTransportResponseMarshaller = Depends()
    ):
        self.unmarshaller = unmarshaller
        self.candidate_sources: List[SimilarTagCandidateSource] = [
            sample_candidate_source,
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

        # TODO: candidate hydration
        # TODO: candidate filtering
        # TODO: candidate ranking
        # TODO: candidate selection

        response: SimilarTagResponse = self.domain_marshaller.marshal(request, flattened_candidates)

        return self.transport_marshaller.marshal(response)
