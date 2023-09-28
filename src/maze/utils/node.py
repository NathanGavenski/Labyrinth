"""Node class for DFS algorithm."""
from functools import lru_cache
import logging
import random
from typing import List, Tuple

try:
    from typing_extensions import Self
except ImportError:
    from typing import Self

import numpy as np


class Node:
    """Node class."""

    def __init__(self, identifier: int, edges: List[Tuple[int, int]]) -> None:
        """Node class for DFS algorithm.

        Args:
            identifier (int): Node identifier.
            edges (List[Tuple[int, int]]): List of edges.
        """
        self.visited = False
        self.identifier = identifier
        self.visited_edges = []
        self.d = []

        random.shuffle(edges)  # So the iterator is random for the DFS
        self.edges = edges
        self.directed_edges = []
        self.walls = []
        self.keep = []
        self.to_delete = []

    def is_visited(self) -> bool:
        """Check if the node was visited."""
        return self.visited

    def add_edge(self, edge: 'Node') -> None:
        """Add an edge to the node.

        Args:
            edge (Node): Node to be added as an edge.
        """
        if isinstance(edge, list):
            self.edges = np.append(self.edges, edge).astype(int)
        else:
            if edge not in self.edges:
                self.edges.append(edge)
        random.shuffle(self.edges)

    def remove_parent(self) -> None:
        """Remove parent from nodes for directed edges."""
        for edge in self.edges:
            if edge not in self.visited_edges:
                msg_log = f"remove_parent:Removing {self.identifier} from {edge.identifier}"
                msg_log += ", previous node on path"
                logging.debug(msg_log)
                self.directed_edges.append(edge)

        self.edges = self.directed_edges
        self.visited_edges = []

    def remove_edge_no_walls(self, edge: 'Node') -> bool:
        """Remove an edge from the node."""
        try:
            self.edge.remove(edge)
            return True
        except ValueError:
            return True

    def remove_edge(self, edge: 'Node') -> None:
        """Remove an edge from the node and add it as a wall."""
        self.edges.remove(edge)
        self.walls.append(edge)

    # FIXME this is clealy worng, it should see if there is a cycle.
    def add_d(self, d: List['Node']) -> None:
        """Add a path to the node if it is not on the list.
        Args:
            d List[Node]: list of nodes.
        """
        if len(d) == len(set(d)) and d not in self.d:
            self.d.append(d)

    def visited_from(self, node: 'Node') -> None:
        """Mark the node as visited from another node.

        Args:
            node (Node): Node that visited the current node.
        """
        self.visited_edges.append(node)

    def set_neighbor(self, neighbor: list) -> None:
        """Set the node neighbors.

        Args:
            neighbor (list): List of neighbors.

        Returns:
            Self: Node object.
        """
        random.shuffle(neighbor)
        self.add_edge(neighbor)

    def get_random_neighbor(self) -> Self:
        """Get a random neighbor from the node.

        Returns:
            Self: Node object.
        """
        unvisted_edges = list(
            set(self.edges).difference(self.visited_edges)
        )
        return random.choice(unvisted_edges)

    def get_neighbors(self) -> List[Self]:
        """Get all the neighbors from the node.

        Returns:
            List[Self]: List of neighbors.
        """
        return self.edges

    @lru_cache(maxsize=156)
    def get_d(self):
        return self.d

    def __len__(self) -> int:
        """Get the number of available edges.

        Returns:
            int: Number of available edges.
        """
        current_available_edges = []
        for node in self.edges:
            if not node.visited:
                current_available_edges.append(node)
        return len(current_available_edges)

    def __getitem__(self, idx: int) -> Self:
        """Get the node from the edge.

        Args:
            idx (int): Index of the edge.

        Returns:
            Self: Node object.
        """
        return self.edges[idx]

    def __eq__(self, other: 'Node') -> bool:
        """Check if the node is equal to another node.

        Args:
            other (Node): Node to be compared.

        Returns:
            bool: True if the nodes are equal, False otherwise.
        """
        return self.identifier == other.identifier

    def __lt__(self, other: 'Node') -> bool:
        """Check if the node is less than another node.

        Args:
            other (Node): Node to be compared.

        Returns:
            bool: True if the node is less than the other node, False otherwise.
        """
        return self.identifier < other.identifier

    def __hash__(self) -> int:
        """Get the node hash.

        Returns:
            int: Node hash.
        """
        return self.identifier

    def __repr__(self) -> str:
        """Get the node representation.

        Returns:
            str: Node representation as a string.
                The node representation is a list of edges.
                Ex: [edge_identifier, edge_identifier, ...]
        """
        return f'{self.identifier}'

    def __str__(self) -> str:
        """Get the node string representation.

        Returns:
            str: Node string representation.
                The node string representation is a dictionary of edges.
                Ex: {node_identifier: [edge_identifier, edge_identifier, ...]}
        """
        current_available_edges = []
        for node in self.edges:
            if not node.visited:
                current_available_edges.append(node)

        return f'{self.identifier}: {[edge.identifier for edge in current_available_edges]}'
