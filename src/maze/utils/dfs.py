"""Depth first search algorithm for the maze generation."""
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

from .node import Node


class DFS:
    """Depth first search algorithm"""

    def __init__(
            self,
            graph: List[Tuple[int, int]],
            shape: Tuple[int, int],
            start: int = None,
            end: int = None
    ) -> None:
        """Depth first search algorithm for the maze generation.

        Args:
            graph (List[Tuple[int, int]]): Graph structure for the maze, containing the edges.
                Edges are represented by a tuple of two integers (x, y) coordinates.
            shape (Tuple[int, int]): Size of the map (width, height)
            start (int, optional): Where the maze starts. Defaults to None.
            end (int, optional): Where the maze ends. Defaults to None.
        """

        self.start = 0 if start is None else start
        self.end = shape[0] * shape[1] - 1 if end is None else end
        self.key = None
        self.door = None
        self.key_and_door = False
        self.shape = shape
        self.graph = graph
        self.nodes = []
        self.paths = []
        self.path = []
        self.reset()

    def reset(self) -> None:
        """Reset the Node graph."""
        del self.nodes

        edges_dict = defaultdict(list)
        for key, neighbor in self.graph:
            edges_dict[key].append(neighbor)

        self.nodes = []
        for node in range(self.shape[0] * self.shape[1]):
            self.nodes.append(Node(node, []))

        for key, value in edges_dict.items():
            self.nodes[key].set_neighbor(value)

    def set_key_and_door(self, key: int, door: int) -> None:
        """
        Set key and door positions (global) and key_and_door mode.

        Args:
            key (int): Global position for key.
            door (int): Global position for door.
        """
        if isinstance(key, tuple):
            raise ValueError("Key must be a single integer. Be sure to pass the global position.")

        if isinstance(door, tuple):
            raise ValueError("Door must be a single integer. Be sure to pass the global position.")

        self.key = key
        self.door = door
        self.key_and_door = True

    def generate_path(
            self,
            visited: List[Tuple[int, int]],
            start: int = None
    ) -> List[Tuple[int, int]]:
        """Generates a maze-like with DFS.

        Args:
            visited (List[Tuple[int, int]]): list of visited nodes. If first iteration use an 
                empty list (default: empty)
            start (int, optional): where to start the maze. Defaults to None.

        Returns:
            List[int]: List of tuples with all edges that form the paths. To form a maze remove 
                these edges from the env.
        """
        current = self.start if start is None else start

        if start is None:
            self.nodes[current].visited = True

        if current == self.end:
            self.nodes[current].visited = False

        for node in self.nodes[current]:
            node = self.nodes[node]
            if not node.is_visited():
                node.visited_from(current)
                visited.append((current, node.identifier))
                visited = self.generate_path(visited, node.identifier)
        return visited

    def find_paths(self, edges: Dict[int, List[Tuple[int, int]]]) -> List[List[int]]:
        """Discover all possible paths to the goal of the maze.

        Args:
            edges (Dict[int, List[Tuple[int, int]]]): Dict with node identifiers and its neighbors.

        Returns:
            List[List[int]]: A list of list of all the nodes that take the agent to its goal.
        """
        nodes_dict = {x: Node(x, []) for x in edges.keys()}

        for x_position, y_position in edges.items():
            for node in y_position:
                nodes_dict[x_position].add_edge(nodes_dict[node])

        for node in nodes_dict.values():
            print(node)
        exit()

        self.path = []
        if not self.key_and_door:
            self._find_paths(set(), nodes_dict, nodes_dict[self.start], [], self.end)
        else:
            self._find_paths(set(), nodes_dict, nodes_dict[self.start], [], self.key)
            key, self.path = self.path, []
            self._find_paths(set(), nodes_dict, nodes_dict[self.key], [], self.door)
            door, self.path = self.path, []
            self._find_paths(set(), nodes_dict, nodes_dict[self.door], [], self.end)
            end, self.path = self.path, []

            for path in key:
                for _path in door:
                    for __path in end:
                        paths = np.append(path[:-1], _path[:-1])
                        self.path.append(np.append(paths, __path).tolist())

        return self.path

    # FIXME This is clearly wrong need to remake
    def _find_paths(self, visited, graph, node, path, end) -> None:
        """Auxiliary recursion function for the find_paths()."""
        path = list(tuple(path))

        if node.identifier == end:
            path.append(node.identifier)
            self.path.append(path)
            return

        if node.identifier not in visited:
            path.append(node.identifier)
            visited.add(node.identifier)
            for neighbor in node:
                self._find_paths(visited, graph, neighbor, path, end)
