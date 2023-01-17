"""Python script to benchmark performance.

We can load a dataset of inputs and expected outputs, as well as a rate-limit.

This doubles as a smoke-test and as an end-to-end benchmarking test. Probably not advisable for production, but kills
two birds with one stone. We can define a set of input and output tags and loops over them to do a performance test.

We can evaluate the quality of a single pass over the dataset, and we can measure the end-to-end latency of the service.

For metric evaluation:
 - We don't have a good precision-recall definition, since true-negatives are undefined...
 - Maybe for now, we can use Jaccard similarity between the results.

"""
import argparse
import os
import urllib.parse
import requests
import time

from tqdm import tqdm

import numpy as np
import pandas as pd


# needed for calling localhost
os.environ['NO_PROXY'] = '127.0.0.1'


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input_data_path", help="path to input tags and expected output tags"
    )
    argparser.add_argument(
        "--qps", type=int, default=50, help="average queries per sec (hz)"
    )
    argparser.add_argument(
        "--max_requests", type=int, required=True, help="max requests for benchmark test"
    )
    argparser.add_argument(
        "--port", type=int, default=3000, help="localhost port to send requests to"
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="set for verbose output",
    )
    argparser.add_argument(
        "--downsample",
        type=int,
        help="Randomly down-sampled rows to evaluate. If None, evaluate all rows"
    )
    return argparser.parse_args()


def evaluate_latencies(elapsed_times):
    print(f"""
Latency (ms)

    p50: {np.percentile(elapsed_times, 50) * 1000:.02f}ms
    p75: {np.percentile(elapsed_times, 75) * 1000:.02f}ms
    p90: {np.percentile(elapsed_times, 90) * 1000:.02f}ms
    p95: {np.percentile(elapsed_times, 95) * 1000:.02f}ms
    p99: {np.percentile(elapsed_times, 99) * 1000:.02f}ms
""")


def evaluate_metrics(test_df):
    assert "result_tags" in test_df
    assert "output_tags" in test_df
    jaccard = []
    for result_tags, output_tags in zip(test_df.result_tags, test_df.output_tags):
        a, b = set(result_tags), set(output_tags)

        jaccard.append(len(a & b) / len(a | b))

    print(f"""
Metrics:
 - Average jaccard: {sum(jaccard) / len(jaccard):.02f}
 - think of better metrics...
""")


def main():
    args = parse_args()

    test_df = pd.read_json(args.input_data_path, orient="records")

    if args.downsample:
        test_df = test_df.sample(args.downsample)

    num_test_rows = len(test_df)
    test_df['encoded_inputs'] = test_df.input_tags \
        .map(lambda tags: ','.join(tags)) \
        .map(urllib.parse.quote)

    if args.verbose:
        print(f'Test row count: {num_test_rows}')
        print(test_df.head())

    feed_data = list(zip(test_df.encoded_inputs, test_df.output_tags))

    # warm-up
    result_tags = []
    for input_tags, output_tags in tqdm(feed_data):
        resp = requests.get(f"http://127.0.0.1:{args.port}/get_similar_tags?tags={input_tags}&limit=5")
        tags = [tag["tag"] for tag in resp.json()]
        result_tags.append(tags)
        time.sleep(0.1)  # don't slam the service during warm-up

    test_df["result_tags"] = result_tags

    # with tempo!
    count = 0
    elapsed_times = []
    while count < args.max_requests:
        input_tags, _ = feed_data[count % num_test_rows]
        start_time = time.perf_counter()
        requests.get(f"http://127.0.0.1:{args.port}/get_similar_tags?tags={input_tags}&limit=5")
        elapsed_time = time.perf_counter() - start_time
        elapsed_times.append(elapsed_time)
        time.sleep(1 / args.qps)
        count += 1

    evaluate_latencies(elapsed_times)
    evaluate_metrics(test_df)


if __name__ == "__main__":
    main()
