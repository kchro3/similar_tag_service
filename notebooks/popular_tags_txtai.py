import fire
import logging
import pandas as pd
import pickle

from collections import Counter
from txtai.embeddings import Embeddings
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def preprocess_documents(df, max_tags):
    documents = []
    metadata_by_docid = {}

    tags = []
    for _, row in df.iterrows():
        for tag in row['tags'] + row['root_tags']:
            tags.append(tag)

    for i, (tag, count) in enumerate(Counter(tags).most_common(max_tags)):
        documents.append((i, tag, None))
        metadata_by_docid[i] = {
            'tag': tag,
            'count': count
        }

    return documents, metadata_by_docid


def main(
    input_data_path: str,
    output_embedding_path: str,
    output_metadata_path: str,
    embedding_path: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 1000,
    max_tags: int = 10000,
):
    logger.info(f"Loading embeddings from {embedding_path}...")
    embeddings = Embeddings({
        "path": embedding_path
    })

    logger.info("Preprocessing dataset into indexable documents...")
    post_df = pd.read_json(input_data_path, orient="records")

    documents, metadata_by_docid = preprocess_documents(post_df, max_tags)

    logger.info(f"Indexing {len(documents)} embeddings...")
    for i in tqdm(range(0, len(documents), batch_size)):
        embeddings.upsert(documents[i:i + batch_size])

    logger.info("Exporting files...")
    embeddings.save(output_embedding_path)
    with open(output_metadata_path, 'wb') as w:
        pickle.dump(metadata_by_docid, w)

    logger.info("done.")


if __name__ == "__main__":
    fire.Fire(main)
