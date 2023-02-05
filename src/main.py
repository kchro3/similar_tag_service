"""Fast API docs

"""
from typing import Union

from dependency_injector.wiring import inject
from fastapi import FastAPI

from src.container import Container
from src.pipeline import SimilarTagRecommendationPipeline


container = Container()
pipeline: SimilarTagRecommendationPipeline = container.pipeline()
app = FastAPI()


@app.get("/get_similar_tags")
@inject
async def get_similar_tags(
    tags: Union[str, None] = None,
    limit: Union[int, None] = None
):
    return await pipeline.execute({
        "tags": tags,
        "limit": limit
    })
