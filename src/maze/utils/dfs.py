from collections import defaultdict
from typing import List, Tuple

import numpy as np

from .node import Node
from .utils import recursionLimit

class DFS:
    def __init__(self, graph: list, shape: tuple, start: int = None, end: int = None):
        '''
        Depth first search algorithm for the maze generation.

        graph : list = list of edges for each cell (ex: [(0, 1), (0, 5)])
        shape : tuple = size of the map (width, height)
        start : int = absolute position for the start (default: 0)
        end : int = absolute position for the end (default: the last cell - up left most cell)
        '''
        self.start = 0 if start is None else start
        self.end = shape[0] * shape[1] - 1 if end is None else end
        self.key = None
        self.door = None
        self.key_and_door = False
        self.shape = shape
        self.graph = graph
        self.nodes = []
        self.paths = []
        self.reset()
        
    def reset(self) -> None:
        """
        Reset the Node graph.

        Returns:
            None
        """
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
        Set key and door positions (global) and key_and_door mode

        Args:
            key: int = global position for key
            door: int = global position for door

        Returns:
            None
        """
        self.key = key
        self.door = door
        self.key_and_door = True

    def generate_path(self, visited: List[Tuple[int, int]], start: int = None) -> List[int]:
        """
        Generates a maze-like with DFS.

        Args:
            visited: List[int] = list of visited nodes (default: empty)
            start: int = where to start the maze (default: None - self.start)

        Returns:
            a list of tuples with all edges that form the paths. To form a maze remove these edges from the env.
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

    def find_paths(self, edges: defaultdict) -> List[List[int]]:
        """
        Discover all possible paths to the goal of the maze.

        Args:
            edges: defaultdict = Dict with the node identifier and its neighbors.

        Returns:
             a list of list of all the nodes that take the agent to its goal.
        """
        nodes_dict = {x: Node(x, []) for x in edges.keys()}

        for x, y in edges.items():
            for node in y:
                nodes_dict[x].add_edge(nodes_dict[node])
            
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
                        self.path.append(np.append(paths, __path))

        return self.path
            
    def _find_paths(self, visited, graph, node, path, end) -> None:
        '''
        Auxiliary recursion function for the find_paths().
        '''
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