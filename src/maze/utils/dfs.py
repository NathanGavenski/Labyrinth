"""Depth first search algorithm for the maze generation."""
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict

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
                node.visited = True
                visited.append((current, node.identifier))
                visited = self.generate_path(visited, node.identifier)
        return visited

    def find_paths(
        self,
        edges: Dict[int, List[Tuple[int, int]]],
        shortest: True,
    ) -> List[List[Node]]:
        """Discover all possible paths to the goal of the maze.

        Args:
            edges (Dict[int, List[Tuple[int, int]]]): Dict with node identifiers and its neighbors.

        Returns:
            List[List[int]]: A list of list of all the nodes that take the agent to its goal.
        """
        if not self.key_and_door:
            if shortest:
                path = self.find_path(edges, self.start, self.end, True)
                if len(path[self.end].d) > 0 and isinstance(path[self.end].d[0], list):
                    return path[self.end].d
                else:
                    return [path[self.end].d]
            else:
                path = self.find_path(edges, self.start, self.end, False)
                while True:
                    for node in path.values():
                        node.visited = False
                    self.update = False
                    self._find_all_paths(path[self.end], path[self.start])
                    if not self.update:
                        break
                if isinstance(path[self.end].d[0], list):
                    return path[self.end].d
                else:
                    return [path[self.end].d]

        start_key_nodes = self.find_path(edges, self.start, self.key, True)
        key_door_nodes = self.find_path(edges, self.key, self.door, True)
        door_end_nodes = self.find_path(edges, self.door, self.end, True)
        start_key_path = start_key_nodes[self.key].d
        key_door_path = key_door_nodes[self.door].d
        door_end_path = door_end_nodes[self.end].d
        path = start_key_path + key_door_path[1:] + door_end_path[1:]
        return [path]

    def find_path(
        self,
        edges: Dict[int, List[Tuple[int, int]]],
        start: int,
        finish: int,
        early_stop: bool
    ) -> Dict[int, Node]:
        """Function for finding viable path.

        Args:
            graph (Dict[int, Node]): graph with edges.
            start (int): starting node.
            finish (int): finishing node,
            early_stop (bool): whether it should keep on finding paths.

        Returns:
            path (List[Node]): list of path(s) from start to finish.
        """
        path = {x: Node(x, []) for x in range(max(edges) + 1)}

        for x_position, y_position in edges.items():
            for node in y_position:
                path[x_position].add_edge(path[node])

        self._find_path(path[start].d, path[start], path[finish], early_stop)
        return path

    def _find_path(
        self,
        visited: list[Node],
        node: Node,
        end: Node,
        early_stop: bool = False
    ) -> None:
        """Auxiliary function for finding path from start to goal. Default non-stop
        DFS implementation. It doesn't stop once it find the end, so we can finding
        all possible paths afterwards.

        Args:
            visited List[Node]: list of all visited nodes (starts empty).
            node Node: current node.
            end Node: end Node.
        """
        node.visited = True
        node.d += visited
        node.d.append(node)

        if not node.identifier == end.identifier:
            for edge in node.edges:
                edge.visited_from(node)
                if not edge.is_visited():
                    self._find_path(node.d, edge, end)
        elif early_stop:
            return

    def _find_all_paths(
        self,
        node: Node,
        start: Node
    ) -> None:
        """Function for finding all paths from start to goal. This function works
        backwards from the graph. It searches for possible paths going from the
        end node until it doesn't find any more forks on the graphs. From that,
        it works forward updating all nodes until the end.

        Args:
            node Node: node to start looking for all possible paths. Should
                start with the goal node.
        """
        node.visited = True
        not_part_of = []

        if node.identifier == start.identifier:
            return

        for edge in node.visited_edges:
            if not isinstance(edge.d[0], list):
                edge.d = [edge.d]
            if not isinstance(node.d[0], list):
                node.d = [node.d]

            if not set(tuple(x) for x in edge.d).issubset(tuple(x[:-1]) for x in node.d):
                self.update = True
                not_part_of.append(edge)

            if not edge.visited:
                self._find_all_paths(edge, start)

        for edge in not_part_of:
            for d in edge.get_d():
                node.add_d(d + [node])
