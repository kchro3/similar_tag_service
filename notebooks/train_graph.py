"""Experimenting with the NetworkX library.

Goal: to create a bipartite graph of blog_urls to tags.

"""
import argparse
import pickle
import logging

import networkx as nx
import pandas as pd

from tqdm import tqdm
from notebooks.model.bipartite_graph import WeightedBipartiteGraph


logging.basicConfig()
logger = logging.getLogger(__name__)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input_data_path", help="path to training JSONL data"
    )
    argparser.add_argument(
        "--export_graph_path", help="path to export graph"
    )
    argparser.add_argument(
        "--export_precomputed_path", help="path to export precomputed 2-hop edges"
    )
    argparser.add_argument(
        "--verbose", type=bool, default=True, help="set for verbose output"
    )
    return argparser.parse_args()


def post_df_to_graph(post_df: pd.DataFrame):
    """
    NetworkX bipartite graph where 0 represents some key and 1 represents tags.

    Edge weights are the counts between key & tags
    """
    tag_counts_by_key = {}
    unique_keys = set()
    unique_tags = set()
    for i, row in post_df.iterrows():
        key = f'post_{row["post_id"]}'
        unique_keys.add(key)
        for tag in row["tags"]:
            unique_tags.add(tag)
            if (key, tag) not in tag_counts_by_key:
                tag_counts_by_key[(key, tag)] = 1
            else:
                tag_counts_by_key[(key, tag)] += 1

    graph = nx.Graph()
    graph.add_nodes_from(unique_keys, bipartite=0)
    graph.add_nodes_from(unique_tags, bipartite=1)

    weighted_edges = [
        (key, tag, count)
        for ((key, tag), count) in tag_counts_by_key.items()
    ]

    graph.add_weighted_edges_from(weighted_edges)

    logger.debug(f"number of unique keys: {len(unique_keys)}")
    logger.debug(f"number of unique tags: {len(unique_tags)}")
    logger.debug(f"number of rows: {sum(tag_counts_by_key.values())}")

    return graph


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    traintest_post_df = pd.read_json(args.input_data_path, orient="records")

    # We want to transpose the dataset such that each row represents a (key, tag)
    graph = post_df_to_graph(traintest_post_df)

    # Precompute the two-hop edges
    logger.debug("Precomputing the two-hop edges...")
    bipartite = WeightedBipartiteGraph(graph)
    precomputed = dict()
    for tag in tqdm(bipartite.get_top_nodes()):
        precomputed[tag] = bipartite.weighted_two_hop(tag)

    logger.debug(f"Exporting {args.export_graph_path}...")
    nx.write_graphml(graph, args.export_graph_path)
    logger.debug(f"Exporting {args.export_precomputed_path}...")
    with open(args.export_precomputed_path, 'wb') as w:
        pickle.dump(precomputed, w)


if __name__ == "__main__":
    main()
