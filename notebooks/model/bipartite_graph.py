from typing import List, Tuple

import networkx as nx


class WeightedBipartiteGraph:
    """
    This was going to be part of the core library, but it wasn't performant enough,
    so we can precompute the 2-hops offline.
    """

    def __init__(self, G):
        self.G = G
        self.bipartite = nx.get_node_attributes(G, "bipartite")

        # check that graph is bipartite so that the algorithm will work as expected
        assert nx.bipartite.is_bipartite(self.G), "Graph is not bipartite"

    def get_weight(self, a, b) -> int:
        return self.G[a][b]["weight"]

    def weighted_hop(self, node: str, k: int = 5) -> List[Tuple[str, int]]:
        if node not in self.G:
            return []
        else:
            weighted_neighbors = []

            for n in self.G[node]:
                weighted_neighbors.append((n, self.get_weight(node, n)))

            return sorted(weighted_neighbors, key=lambda x: x[1], reverse=True)[:k]

    def weighted_two_hop(self, node: str, k: int = 5) -> List[Tuple[str, int]]:
        neighbors_and_weights = {}
        for n, w in self.weighted_hop(node, k):
            for nn, ww in self.weighted_hop(n, k):
                if node != nn:
                    if nn not in neighbors_and_weights:
                        neighbors_and_weights[nn] = w+ww
                    else:
                        neighbors_and_weights[nn] += w+ww

        return sorted(
            neighbors_and_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

    def get_top_nodes(self) -> List[int]:
        top_nodes = []
        for node in self.G:
            if self.bipartite[node] == 1:
                top_nodes.append(node)
        return top_nodes

    def get_bottom_nodes(self) -> List[int]:
        bottom_nodes = []
        for node in self.G:
            if self.bipartite[node] == 0:
                bottom_nodes.append(node)
        return bottom_nodes
