import fire
import logging
import pandas as pd
import pickle

from gensim.parsing.preprocessing import *
from txtai.embeddings import Embeddings
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def strip_newlines(x):
    return x.replace('\\n', ' ')


FILTERS = [
    strip_newlines,
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    remove_stopwords,
]


def preprocess_documents(df, max_chars=100):
    """Preprocesses text from the title and tags.

    I had a few observations:
    - the title of the reblogged posts appeared to be the body of the original post
    - the embeddings would take a long time if the text was not truncated
    - the text needed to be purged of html tags, newlines, and other punctuations

    The goal for this section is to do a semantic search for a post, and then suggest
    its tags as similar tags. Therefore, we want to also persist the lookup table of
    post ID to tags.

    :param df:
    :param max_chars:
    :return: documents are a list of tuples, and metadata is a dict of dicts.
    """
    preprocessed_title = df.title.map(lambda x: preprocess_string(x, FILTERS))
    df['doc'] = list(map(lambda x: ' '.join(x[0] + x[1]), zip(preprocessed_title, df.root_tags)))

    documents = []
    metadata_by_docid = {}
    for i, row in df.iterrows():
        documents.append((i, row['doc'][:max_chars], None))
        metadata_by_docid[i] = {
            'doc': row['doc'],
            'tags': row['tags'] + row['root_tags']
        }

    return documents, metadata_by_docid


def main(
    input_data_path: str,
    output_embedding_path: str,
    output_metadata_path: str,
    embedding_path: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 1000,
    downsample=None,
):
    logger.info(f"Loading embeddings from {embedding_path}...")
    embeddings = Embeddings({
        "path": embedding_path
    })

    logger.info("Preprocessing dataset into indexable documents...")
    post_df = pd.read_json(input_data_path, orient="records")
    if downsample is not None:
        # for debugging
        post_df = post_df.sample(downsample)

    documents, metadata_by_docid = preprocess_documents(post_df)

    logger.info(f"Indexing {len(documents)} embeddings from {input_data_path}...")
    for i in tqdm(range(0, len(documents), batch_size)):
        embeddings.upsert(documents[i:i + batch_size])

    logger.info("Exporting files...")
    embeddings.save(output_embedding_path)
    with open(output_metadata_path, 'wb') as w:
        pickle.dump(metadata_by_docid, w)

    logger.info("done.")


if __name__ == "__main__":
    fire.Fire(main)
