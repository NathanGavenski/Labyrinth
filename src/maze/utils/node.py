"""Node class for DFS algorithm."""
import random
from typing import List, Tuple

try:
    from typing_extensions import Self
except ImportError:
    from typing import Self

import numpy as np


class Node:
    """Node class."""
    def __init__(self, identifier:int, edges: List[Tuple[int, int]]) -> None:
        """Node class for DFS algorithm.

        Args:
            identifier (int): Node identifier.
            edges (List[Tuple[int, int]]): List of edges.
        """
        self.visited = False
        self.identifier = identifier
        self.visited_edges = []

        random.shuffle(edges) # So the iterator is random for the DFS
        self.edges = edges

    def is_visited(self) -> bool:
        """Check if the node was visited."""
        return self.visited

    def add_edge(self, edge: 'Node') -> None:
        """Add an edge to the node.

        Args:
            edge (Node): Node to be added as an edge.
        """
        self.edges.append(edge)

    def visited_from(self, node: 'Node') -> None:
        """Mark the node as visited from another node.

        Args:
            node (Node): Node that visited the current node.
        """
        self.visited = True
        if isinstance(node, int):
            self.visited_edges.append(node)
        else:
            self.visited_edges.append(node.identifier)

    def set_neighbor(self, neighbor:list) -> Self:
        """Set the node neighbors.

        Args:
            neighbor (list): List of neighbors.

        Returns:
            Self: Node object.
        """
        random.shuffle(neighbor)
        self.edges = np.array(neighbor)
        return self

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

    def __getitem__(self, idx:int) -> Self:
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
        return f'{list(self.edges)}'

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
