"""Depth first search algorithm for the maze generation."""
from collections import defaultdict
import logging
from typing import List, Tuple, Dict
import random

from .node import Node


class DFS:
    """Depth first search algorithm"""

    def __init__(
            self,
            graph: List[Tuple[int, int]],
            shape: Tuple[int, int],
            start: int = None,
            end: int = None,
            random_amount: int = 0
    ) -> None:
        """Depth first search algorithm for the maze generation.

        Args:
            graph (List[Tuple[int, int]]): Graph structure for the maze, containing the edges.
                Edges are represented by a tuple of two integers (x, y) coordinates.
            shape (Tuple[int, int]): Size of the map (width, height)
            start (int, optional): Where the maze starts. Defaults to None.
            end (int, optional): Where the maze ends. Defaults to None.
            random_amount (int, optional): How likely it should allow a edge to stay.
                Defaults to 0 (it does not happen).
        """

        self.start = 0 if start is None else start
        self.end = shape[0] * shape[1] - 1 if end is None else end
        self.key = None
        self.door = None
        self.key_and_door = False
        self.shape = shape
        self.graph = graph
        self.nodes = []
        self.update = None
        self.paths = []
        self.path = []
        self.random_amount = random_amount
        self.reset()

    def reset(self) -> None:
        """Reset the Node graph."""
        del self.nodes

        edges_dict = defaultdict(list)
        for key, neighbor in self.graph.items():
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

    def convert_graph(self) -> None:
        """Convert a start and end into a graph."""
        path = {x: Node(x, []) for x in range(self.end + 1)}

        for edges in self.graph.values():
            for (x, edge) in edges:
                path[x].add_edge(path[edge])

        if logging.getLogger().level == logging.DEBUG:
            for node in path.values():
                logging.debug(f"generate_path:Node {node.identifier} with edges {node.edges}")
        self.graph = path


    def generate_path(
            self,
            min_paths: int = None,
            max_paths: int = None,
    ) -> List[Node]:
        """Generates a maze-like with DFS.

        Args:
            min_paths (int, optional): whether the maze has to have more than one path.
                Defaults to None (no paths required).

        Returns:
            graph (List[Node]): List of nodes. These nodes have a list for edges and walls.
        """
        path = {x: Node(x, []) for x in range(max(self.graph.keys()) + 1)}

        for edges in self.graph.values():
            for (x, edge) in edges:
                path[x].add_edge(path[edge])

        logging.debug([
            f"generate_path:Node {node.identifier} with edges {node.edges}" \
            for node in path.values()
        ])

        self._generate_path(path[self.start], path[self.start], path[self.end])

        logging.debug([
            f"generate_path:Node {node.identifier} with edges {node.edges}" + \
            f"and visitors {node.visited_edges}" \
            for node in path.values()
        ])


        for node in path.values():
            node.remove_parent()
        path[self.end].edges = []

        logging.debug([
            f"generate_path:Node {node.identifier} with edges {node.edges}\n" \
            for node in path.values()
        ])

        if self.find_loop(path):
            logging.error("Found a loop")

        if min_paths is not None or max_paths is not None:
            path = self.find_path(path, self.start, self.end, False)

            while True:
                for node in path.values():
                    node.visited = False

                self.update = False
                self._find_all_paths(path[self.end], path[self.start])
                if not self.update:
                    break

                if min_paths is not None and len(path[self.end].d) > min_paths:
                    break

                if max_paths is not None and len(path[self.end].d) > max_paths:
                    logging.debug(
                        "generate_path:Generated more paths than allowed: %i",
                        len(path[self.end].d)
                    )
                    self.graph = self.generate_path(min_paths, max_paths)
                    return self.graph

                if max_paths is not None and len(path[self.end].d) <= max_paths:
                    break

            logging.debug(f"generate_path:{path[self.end].d} solutions found for maze")

            if min_paths is not None and min_paths > len(path[self.end].d):
                self.graph = self.generate_path(min_paths, max_paths)
                return self.graph

        self.graph = path

        logging.debug([
            "generate_path:" +
            f"Node {node.identifier} with " +
            f"edges {node.directed_edges} and " +
            f"walls {node.walls}"
            for node in self.graph.values()
        ])

        return path

    def _generate_path(
        self,
        node: Node,
        start: Node,
        end: Node,
    ) -> None:
        """Recursive function to create maze. It performs a DFS algorithm where it visits each 
        node once. If the node was already visited, the edge has a random chance to not be removed.
        It does not delete edges that were already visited to not cause no solutions in the maze.

        Args:
            node (Node): current node.
            start (Node): starting node.
            end (Node): ending node.
        """
        logging.debug(f"Visiting node {node.identifier}")

        if node.visited:
            return

        node.visited = True
        if node == end:
            return

        for edge in node.edges:
            logging.debug(f"_generate_path:Node {node.identifier} with edge {edge.identifier}")
            if not edge.visited:
                node.keep.append(edge)
                edge.visited_from(node)
                self._generate_path(edge, start, end)
            elif node in edge.to_delete:
                logging.debug(
                    "_generate_path:Node %i already in %i to_delete",
                    node.identifier,
                    edge.identifier
                )
            elif node not in edge.keep and random.randint(0, 100) > self.random_amount:
                logging.debug(
                    "_generate_path:Node %i, removing edge %i",
                    node.identifier,
                    edge.identifier
                )
                node.to_delete.append(edge)

        for edge in node.to_delete:
            node.remove_edge(edge)
            edge.remove_edge(node)

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
                return [path[self.end].d]
            path = self.find_path(self.graph, self.start, self.end, False)
            logging.debug(f"find_paths:First path found: {path[self.end].d}")
            logging.debug([
                f"find_paths:Node {node.identifier} with edges {node.edges}"
                for node in path.values()
            ])

            while True:
                for node in path.values():
                    node.visited = False
                self.update = False
                self._find_all_paths(path[self.end], path[self.start])
                if not self.update:
                    break

            logging.debug(f"generate_path:{path[self.end].d} solutions found for maze")
            if isinstance(path[self.end].d[0], list):
                return path[self.end].d
            return [path[self.end].d]

        start_key_nodes = self.find_path(edges, self.start, self.key, True)
        key_door_nodes = self.find_path(edges, self.key, self.door, True)
        door_end_nodes = self.find_path(edges, self.door, self.end, True)
        start_key_path = start_key_nodes[self.key].d
        key_door_path = key_door_nodes[self.door].d
        door_end_path = door_end_nodes[self.end].d
        path = start_key_path + key_door_path[1:] + door_end_path[1:]
        return [path]

    def find_loop(self, graph: Dict[int, List[Node]]) -> bool:
        """Detect whether a loop exists in a graph.

        Args:
            graph (dict[int, list[Node]]): graph nodes.

        Returns:
            loop_exists (bool): True if there is a loop.
        """
        for node in graph.values():
            for edge in node.edges:
                if node in graph[edge.identifier].edges:
                    logging.debug(
                        "Node %i found in %i",
                        edge.identifier,
                        node.identifier
                    )
                    return True
        return False

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
        if isinstance(list(edges.values())[0], Node):
            path = {node.identifier: Node(node.identifier, []) for node in edges.values()}
            for node in edges.values():
                for edge in node.edges:
                    path[node.identifier].add_edge(path[edge.identifier])
                for edge in node.directed_edges:
                    path[node.identifier].directed_edges.append(path[edge.identifier])
                for edge in node.walls:
                    path[node.identifier].walls.append(path[edge.identifier])
        else:
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

    def are_all_subsets(
        self,
        sublist: List[Node],
        superlists: List[List[Node]],
        node: Node
    ) -> bool:
        """Tests if all elemets from a sublist is part of a superlist.

        Args:
            sublist (list[Node]): sublist elements.
            superlists (list[list[Node]]): all superlists.
            node (Node): node to check if it is in the sublist.

        Returns:
            is_part (bool): if it is a sublist from the superlists.
        """
        is_subset = []
        for superlist in superlists:
            is_subset.append(set(sublist) == set(superlist[:-1]) or node in sublist)
        return any(is_subset)

    def are_all_lists_subsets(
        self,
        sublists: List[List[Node]],
        superlists: List[List[Node]],
        node: Node
    ) -> bool:
        """Check if all sublists are part of a superlist.

        Args:
            sublists (list[list[Node]]): list of lists of elements.
            superlists (list[list[Node]]): list of lists of elements.
            node (Node): node element.

        Returns:
            is_part (bool) if all sublists are part of the superlists.
        """
        return all(self.are_all_subsets(sublist, superlists, node) for sublist in sublists)

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
        logging.debug(f"_find_all_paths: Visiting node {node.identifier}")
        not_part_of = []

        if node.visited:
            return

        node.visited = True
        if node.identifier == start.identifier:
            return

        for edge in node.visited_edges:
            if not isinstance(edge.d[0], list):
                edge.d = [edge.d]
            if not isinstance(node.d[0], list):
                node.d = [node.d]

            is_part = self.are_all_lists_subsets(edge.d, node.d, node)
            log_msg = f"_find_all_paths:Comparing {edge.identifier} {edge.d} "
            log_msg += f"with {node.identifier} {node.d} "
            log_msg += f"({is_part})"
            logging.debug(log_msg)
            if not is_part:
                self.update = True
                not_part_of.append(edge)

                for d in edge.get_d():
                    logging.debug(f"Adding {d} to {node.identifier}")
                    logging.debug([
                        f"{node.identifier}: {path} ({path == d})"
                        for path in node.get_d()
                    ])
                    node.add_d(d + [node])

            if not edge.visited:
                self._find_all_paths(edge, start)
