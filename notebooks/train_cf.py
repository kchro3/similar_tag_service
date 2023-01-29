"""Experimenting with the Surprise (https://surpriselib.com/) Open Source collaborative filtering library.


"""
import pandas as pd
from surprise import *
from surprise.dump import dump
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import argparse
from collections import defaultdict


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input_data_path", help="path to training JSONL data"
    )
    argparser.add_argument(
        "--model_type", help="type of model"
    )
    argparser.add_argument(
        "--export_model_path", help="path to export model"
    )
    argparser.add_argument(
        "--min_tag_count", type=int, default=2, help="min threshold for number of times a blog url has used a tag"
    )
    argparser.add_argument(
        "--verbose", type=bool, default=True, help="set for verbose output"
    )
    return argparser.parse_args()


def post_df_to_dataset(post_df: pd.DataFrame, min_tag_count: int, verbose=True) -> Dataset:
    """
    Surprise library expects three fields: users, items, and ratings.

    In our case:
     - "users" can be the post_id
     - "items" can be the tags
     - "ratings" can be the number of times a post_url contains a tag

    :param post_df:
    :return:
    """
    tag_counts_by_post_id = defaultdict(int)
    for i, row in post_df.iterrows():
        post_id = row["post_id"]
        for tag in row["tags"] + row["root_tags"]:
            tag_counts_by_post_id[(post_id, tag)] += 1

    input_df = pd.DataFrame([
        {
            "user": post_id,
            "item": tag,
            "rating": count
        }
        for (post_id, tag), count in tag_counts_by_post_id.items()
        if count >= min_tag_count
    ], columns=["user", "item", "rating"])  # this order is assumed by the Surprise library

    if verbose:
        print(f"number of rows: {len(input_df)}")

    reader = Reader(rating_scale=(input_df["rating"].min(), input_df["rating"].max()))
    data = Dataset.load_from_df(input_df, reader=reader)
    return data


def train(args, trainset):
    sim_options = {
        "user_based": False,  # compute similarities between items
    }

    if args.model_type == "knn":
        algo = KNNBasic(sim_options=sim_options)
    elif args.model_type == "knnz":
        algo = KNNWithZScore(sim_options=sim_options)
    else:
        raise NotImplementedError(f"unknown model type: {args.model_type}")

    stats = cross_validate(algo, trainset, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return algo


def main():
    args = parse_args()
    traintest_post_df = pd.read_json(args.input_data_path, orient="records")

    # We want to transpose the dataset such that each row represents a (post_id, tag)
    # and then we can convert the datasets to conform to Surprise's data model.
    data = post_df_to_dataset(
        traintest_post_df,
        min_tag_count=args.min_tag_count
    )

    algo = train(args, data)

    dump(args.export_model_path, algo=algo)


if __name__ == "__main__":
    main()
