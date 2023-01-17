"""Python script to benchmark performance.

We can load a dataset of inputs and expected outputs, as well as a rate-limit.

This doubles as a smoke-test and as an end-to-end benchmarking test. Probably not advisable for production, but kills
two birds with one stone. We can define a set of input and output tags and loops over them to do a performance test.

We can evaluate the quality of a single pass over the dataset, and we can measure the end-to-end latency of the service.

For metric evaluation:

 - precision is defined as `true-positives / (true-positives + false-positives)`
    - a "true positive" is when the suggested tag is in the expected tags
    - a "false positive" is when the suggested tag is not in the expected tags

 - compute recall as `true-positives / (true-positives + false-negatives)`
    - a "false negative" is when the expected tag is not suggested

"""
import argparse
import requests

import pandas as pd


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input_data_path", help="path to input tags and expected output tags"
    )
    argparser.add_argument(
        "--qps", help="average queries per sec (hz)"
    )
    argparser.add_argument(
        "--port", help="localhost port to send requests to"
    )
    argparser.add_argument(
        "--verbose", type=bool, default=True, help="set for verbose output"
    )
    return argparser.parse_args()


def main():
    args = parse_args()
    test_df = pd.read_json(args.input_data_path, orient="records")
    num_test_rows = len(test_df)

    if args.verbose:
        print(f'Test row count: {num_test_rows}')

    print(test_df.head())

if __name__ == "__main__":
    main()
