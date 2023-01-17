"""Python script to convert our preprocessed Post data into training and test data.

We want to be able to evaluate the quality of tag recommendations, and one option is to measure tag-similarity as
how often tags co-occur in posts. In order to do so, we'll need to split our train and test sets by posts so that we
don't contaminate our train and test dataset.

"""
import argparse
import os

import pandas as pd

from sklearn.model_selection import train_test_split


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input_data_path", help="path to preprocessed JSONL data"
    )
    argparser.add_argument(
        "--output_data_dir", help="dir to output JSONL data"
    )
    argparser.add_argument(
        "--hold_out_pct", type=float, help="percentage of data to hold-out", default=0.2
    )
    argparser.add_argument(
        "--verbose", type=bool, default=True, help="set for verbose output"
    )
    return argparser.parse_args()


def main():
    args = parse_args()
    prep_df = pd.read_json(args.input_data_path, orient="records")
    num_prep_rows = len(prep_df)

    if args.verbose:
        print(f'Preprocessed row count: {num_prep_rows}')

    train_df, test_df = train_test_split(prep_df, test_size=args.hold_out_pct)

    if args.verbose:
        print(f'Train row count: {len(train_df)}')
        print(f'Test row count: {len(test_df)}')

    os.makedirs(args.output_data_dir, exist_ok=True)
    train_df.to_json(f"{args.output_data_dir}/train.jsonl", orient="records")
    test_df.to_json(f"{args.output_data_dir}/test.jsonl", orient="records")


if __name__ == "__main__":
    main()
