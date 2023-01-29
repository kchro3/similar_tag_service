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
import logging

from tqdm import tqdm

import numpy as np
import pandas as pd


# needed for calling localhost
os.environ['NO_PROXY'] = '127.0.0.1'
logging.basicConfig()
logger = logging.getLogger("livetest")


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
        "--skip_eval", action="store_true", help="skip eval and test performance metrics only",
    )
    argparser.add_argument(
        "--downsample",
        type=int,
        help="Randomly down-sampled rows to evaluate. If None, evaluate all rows"
    )
    return argparser.parse_args()


def evaluate_performance(elapsed_times, responses):
    status_codes = []
    agg_num_tags = []
    non_empty = 0
    for response in responses:
        status_codes.append(response.status_code)
        if response.status_code == 200:
            num_tags = len(response.json())
            agg_num_tags.append(num_tags)
            if num_tags > 0:
                non_empty += 1
        else:
            agg_num_tags.append(0)

    success_rate = status_codes.count(200) / len(status_codes)  * 100
    nonempty_rate = non_empty / len(responses) * 100

    logger.info(f"""
Success Rate: {success_rate:.02f}%

Latency (ms)
    p50: {np.percentile(elapsed_times, 50) * 1000:.02f}ms
    p75: {np.percentile(elapsed_times, 75) * 1000:.02f}ms
    p90: {np.percentile(elapsed_times, 90) * 1000:.02f}ms
    p95: {np.percentile(elapsed_times, 95) * 1000:.02f}ms
    p99: {np.percentile(elapsed_times, 99) * 1000:.02f}ms

Non-empty rate: {nonempty_rate:.02f}% ({non_empty} out of {len(responses)})

Response size:
    p10: {np.percentile(agg_num_tags, 10):.02f} results
    p25: {np.percentile(agg_num_tags, 25):.02f} results
    p50: {np.percentile(agg_num_tags, 50):.02f} results
    p75: {np.percentile(agg_num_tags, 75):.02f} results
    p90: {np.percentile(agg_num_tags, 90):.02f} results
""")


def evaluate_metrics(test_df, port):
    logger.debug(f'Test row count: {len(test_df)}')
    logger.debug(test_df.head())

    sample_data = list(zip(test_df.input_tags, test_df.encoded_inputs, test_df.output_tags))

    # warm-up
    result_tags = []
    for input_tags, encoded_tags, output_tags in tqdm(sample_data):
        resp = requests.get(f"http://127.0.0.1:{port}/get_similar_tags?tags={encoded_tags}&limit=5")

        if resp.status_code == 200:
            tags = [tag["tag"] for tag in resp.json()]
            result_tags.append(tags)
        else:
            logger.error(f"input_tags: {input_tags}, encoded_tags: {encoded_tags}, status_code: {resp.status_code}")
            logger.debug(f"traceback: {str(resp.content, 'UTF-8')}")

        time.sleep(0.1)  # don't slam the service during warm-up

    test_df["result_tags"] = result_tags

    jaccard = []
    for input_tags, result_tags, output_tags in zip(test_df.input_tags, test_df.result_tags, test_df.output_tags):
        a, b = set(result_tags), set(output_tags)
        j = len(a & b) / len(a | b)

        if j > 0:
            logger.debug(f"input_tags: {input_tags}")
            logger.debug(f"result_tags: {result_tags}")
            logger.debug(f"output_tags: {output_tags}")
        jaccard.append(j)

    logger.info(f"""
Metrics:
 - Average jaccard: {sum(jaccard) / len(jaccard):.02f}
 - think of better metrics...
""")


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    test_df = pd.read_json(args.input_data_path, orient="records")

    num_test_rows = len(test_df)
    test_df['encoded_inputs'] = test_df.input_tags \
        .map(lambda tags: ','.join(tags)) \
        .map(urllib.parse.quote)

    if not args.skip_eval:
        test_sample_df = test_df.sample(args.downsample)
        evaluate_metrics(test_sample_df, args.port)

    # with tempo!
    count = 0
    elapsed_times = []
    responses = []
    feed_data = list(zip(test_df.input_tags, test_df.encoded_inputs, test_df.output_tags))
    while count < args.max_requests:
        _, encoded_tags, _ = feed_data[count % num_test_rows]
        start_time = time.perf_counter()
        response = requests.get(f"http://127.0.0.1:{args.port}/get_similar_tags?tags={encoded_tags}&limit=5")
        elapsed_time = time.perf_counter() - start_time
        elapsed_times.append(elapsed_time)
        responses.append(response)

        time.sleep(1 / args.qps)
        count += 1

    evaluate_performance(elapsed_times, responses)


if __name__ == "__main__":
    main()
