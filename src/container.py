import yaml

from dependency_injector import containers, providers

from src.candidate_source.poptags_ann_candidate_source import PopularTagsANNSimilarTagCandidateSource
from src.candidate_source.surprise_candidate_source import KNNSimilarTagCandidateSource
from src.embeddings.txtai import TxtaiEmbeddings
from src.marshaller.domain_response import SimilarTagDomainResponseMarshaller
from src.marshaller.request import SimilarTagRequestUnmarshaller
from src.marshaller.transport_response import SimilarTagTransportResponseMarshaller
from src.pipeline import SimilarTagRecommendationPipeline

from src.scorer.cosinesim_scorer import CosineSimScorer
from src.selector.deduping_selector import DedupingSelector
from src.selector.topk_selector import TopKSelector


class Container(containers.DeclarativeContainer):
    with open("src/config.yml", "r") as f:
        config = yaml.safe_load(f)

    unmarshaller = providers.Singleton(SimilarTagRequestUnmarshaller)
    knn_candidate_source = providers.Singleton(
        KNNSimilarTagCandidateSource,
        knn_path=config["surprise"]["knn_path"]
    )
    txtai_embeddings = providers.Singleton(
        TxtaiEmbeddings,
        model_path=config["txtai"]["model_path"],
        embedding_path=config["txtai"]["embedding_path"],
        metadata_path=config["txtai"]["metadata_path"]
    )
    poptags_candidate_source = providers.Singleton(
        PopularTagsANNSimilarTagCandidateSource,
        embeddings=txtai_embeddings
    )
    cosinesim_scorer = providers.Singleton(
        CosineSimScorer,
        embeddings=txtai_embeddings,
        feature_name=config["score_feature"])
    deduping_selector = providers.Singleton(DedupingSelector)
    topk_selector = providers.Singleton(TopKSelector, feature_name=config["score_feature"])
    domain_marshaller = providers.Singleton(
        SimilarTagDomainResponseMarshaller,
        feature_name=config["score_feature"]
    )
    transport_marshaller = providers.Singleton(SimilarTagTransportResponseMarshaller)

    pipeline = providers.Factory(
        SimilarTagRecommendationPipeline,
        unmarshaller=unmarshaller,
        knn_candidate_source=knn_candidate_source,
        poptags_candidate_source=poptags_candidate_source,
        cosinesim_scorer=cosinesim_scorer,
        deduping_selector=deduping_selector,
        topk_selector=topk_selector,
        domain_marshaller=domain_marshaller,
        transport_marshaller=transport_marshaller
    )
