"""Fast API docs

"""
from typing import Union

from fastapi import FastAPI, Depends

from src.pipeline import SimilarTagRecommendationPipeline

app = FastAPI()


@app.get("/get_similar_tags")
async def get_similar_tags(
    tags: Union[str, None] = None,
    limit: Union[int, None] = None,
    pipeline: SimilarTagRecommendationPipeline = Depends()
):
    return await pipeline.execute({
        "tags": tags,
        "limit": limit
    })
