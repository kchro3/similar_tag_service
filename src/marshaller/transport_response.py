from typing import Mapping, List, Any

from src.model.response import SimilarTagResponse


class SimilarTagTransportResponseMarshaller:
    def marshal(self, response: SimilarTagResponse) -> List[Mapping[str, Any]]:
        return [
            {
                "tag": scored_tag.tag,
                "score": scored_tag.score
            }
            for scored_tag in response.scored_tags
        ]
