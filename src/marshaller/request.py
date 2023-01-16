"""SimilarTagRequestUnmarshaller

Unmarshals the transport request to a domain model of the request.

Here we can do basic input validations.

"""

from src.model.request import SimilarTagRequest


class SimilarTagRequestUnmarshaller:
    def unmarshal(self, params: dict) -> SimilarTagRequest:
        if "tags" not in params:
            raise MissingTagsParam()

        tags = params["tags"].split(',')
        limit = params["limit"]
        if limit <= 0:
            raise InvalidNumTagsException()

        return SimilarTagRequest(tags, limit)


class InvalidNumTagsException(Exception):
    pass


class MissingTagsParam(Exception):
    pass
