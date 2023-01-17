"""Python script to generate inputs and expected output tags.

I don't want to spend too much writing a harness, especially since it will likely change.

The steps are:

 - load a hold-out set of posts

 - for each post, we can randomly select one of more tags as the inputs, and the rest are output candidates
    - we can ignore posts that only have one tag, since they have no co-occurring tags
    - 65% of posts in our full dataset have 2+ tags, excluding root tags, and 85% of posts have 2+ tags, inclusive of
      root tags.

"""
import argparse
import os

import pandas as pd

from sklearn.model_selection import train_test_split
from model.post import Post, PostType


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input_data_path", help="path to hold-out JSONL data"
    )
    argparser.add_argument(
        "--output_data_path", help="path to serialized model"
    )
    argparser.add_argument(
        "--verbose", type=bool, default=True, help="set for verbose output"
    )
    return argparser.parse_args()


def unmarshal_post(record):
    return Post(
        post_id=record["index"],
        blog_url=record["blog_url"],
        post_type=PostType[record["type"].upper()],
        lang=record["lang"],
        is_reblog=(int(record["is_reblog"]) == 1),
        tags=record["tags"],
        root_tags=record["root_tags"]
    )


def prepare_cotag_test_data(df):
    """Prepare a gold set of co-occurring tags.

    :param df: dataset of posts
    :return: zipped list of inputs and output tags
    """
    posts = list(map(unmarshal_post, df.to_dict("records")))
    input_tags = []
    output_tags = []
    for post in posts:
        if len(post.tags) >= 2:
            # For now, keeping it simple by splitting into head & tail.
            head, tail = post.tags[0:1], post.tags[1:]
            input_tags.append(head)
            output_tags.append(tail)

    return pd.DataFrame({
        "input_tags": input_tags,
        "output_tags": output_tags
    })


def main():
    args = parse_args()
    test_df = pd.read_json(args.input_data_path, orient="records")
    num_test_rows = len(test_df)

    if args.verbose:
        print(f'Test row count: {num_test_rows}')

    cotag_df = prepare_cotag_test_data(test_df)

    if args.verbose:
        print(f'cotag row count: {len(cotag_df)}')

    cotag_df.to_json(args.output_data_path, orient="records")


if __name__ == "__main__":
    main()
