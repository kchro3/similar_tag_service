import argparse

import pandas as pd

pd.set_option('display.max_columns', None)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input_data_path", help="path to CSV data"
    )
    argparser.add_argument(
        "--output_data_path", help="path to output CSV data"
    )
    argparser.add_argument(
        "--verbose", type=bool, default=True, help="set for verbose output"
    )
    return argparser.parse_args()


def preprocess_tags(comma_separated_tags):
    if len(comma_separated_tags) == 0:
        return []
    else:
        return comma_separated_tags.split(',')


def main():
    args = parse_args()
    raw_df = pd.read_csv(args.input_data_path)
    num_raw_rows = len(raw_df)

    if args.verbose:
        print(f'Raw row count: {num_raw_rows}')

    prep_df = (
        raw_df
        .reset_index(names="post_id")
        .fillna(value={
            "is_reblog": 0.0,
            "root_tags": ''
        })
    )

    prep_df["tags"] = prep_df["tags"].map(preprocess_tags)
    prep_df["root_tags"] = prep_df["root_tags"].map(preprocess_tags)

    assert num_raw_rows == len(prep_df), "Unexpectedly dropped records"
    prep_df.to_json(args.output_data_path, orient="records")


if __name__ == "__main__":
    main()
