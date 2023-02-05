import fire
import time
import pandas as pd
import os
import urllib.parse
import requests
import logging

from tqdm import tqdm

import pandas as pd


# needed for calling localhost
os.environ['NO_PROXY'] = '127.0.0.1'
logging.basicConfig()
logger = logging.getLogger("top_tags")


def main(
    input_dataset="../../notebooks/data/Trial-supplement-tag-followers.csv",
    output_dataset="../../notebooks/data/top_tags_output.tsv",
    top_k=1000,
    port=8000,
    ms_between_calls=10,
    downsample=100,
):

    raw_df = pd.read_csv(input_dataset)
    top_df = raw_df.head(top_k)
    if downsample is not None:
        top_df = top_df.sample(downsample)

    print(top_df.head())
    top_df['encoded_tag'] = top_df["tag"].map(lambda x: urllib.parse.quote(str(x)))

    # warm-up
    result_tags = []
    for encoded_tag in tqdm(top_df.encoded_tag):
        resp = requests.get(f"http://127.0.0.1:{port}/get_similar_tags?tags={encoded_tag}&limit=5")

        if resp.status_code == 200:
            tags = [f'{tag["tag"]}:{tag["score"]:0.2f}' for tag in resp.json()]
            result_tags.append(tags)
        else:
            logger.debug(f"traceback: {str(resp.content, 'UTF-8')}")
            result_tags.append([])

        time.sleep(ms_between_calls / 1000)  # don't slam the service
    top_df['result_tags'] = result_tags

    logger.info(f"Exporting to {output_dataset}")
    top_df.to_csv(output_dataset, sep='\t')


if __name__ == "__main__":
    fire.Fire(main)
