"""Module to compare graphs."""
import ast
from typing import Any, List, Dict, Tuple
from collections import defaultdict

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import gym

from src.maze.utils.node import Node
from src import maze


class Graph:

    def __init__(self, graphs: List[str], params: Dict[str, Any] = None) -> None:
        """Graph class to compare graphs.

        Args:
            graphs (List[str]): List of paths to the graphs.
            params (Dict[str, Any], optional): Parameters for the maze. Defaults to None.
        """
        self.params = params if params is not None else {}
        self.graphs = self.load_graphs(graphs)
        self.solutions = self.find_solutions(graphs)
        self.height = 1000
        self.width = 600

    def load_graphs(self, graphs_path: List[str]) -> Dict[str, Node]:
        """Load graphs from a list of files.

        Args:
            graphs_path (List[str]): List of paths to the graphs.

        Returns:
            Dict[str, Node]: Dictionary with the graphs.
        """
        graphs = {}
        for graph_path in graphs_path:
            with open(graph_path, encoding="utf-8") as f:
                for line in f:
                    edges, *_ = line.split(";")

            edges = np.array(ast.literal_eval(edges))
            for edge in edges:
                edge.sort()

            graph = {}
            for i in range(edges.max() + 1):
                graph[i] = Node(i, edges=[])

            for node, edge in edges:
                graph[node].add_edge(graph[edge])

            graphs[graph_path] = graph
        return graphs

    def find_solutions(self, graph_paths: List[str]) -> List[int]:
        """Find the solutions for the graphs.

        Args:
            graph_paths (List[str]): List of paths to the graphs to load into the environment.

        Returns:
            List[int]: List of the shortest solutions for each graphs.
        """
        solutions = {}
        env = gym.make("Maze-v0", **self.params)
        for structure in graph_paths:
            env.load(structure)
            graph_solutions = env.solve(mode="all")
            lengths = [len(sublist) for sublist in graph_solutions]
            solutions[structure] = graph_solutions[np.argmin(lengths)]
        return solutions

    def path_similarity(
        self,
        other_graph: Dict[str, Node]
    ) -> Tuple[Dict[str, float], List[int]]:
        """Calculate the similarity between the paths of the graphs.

        Args:
            other_graph (Dict[str, Node]): Graph to compare.

        Returns:
            Tuple[Dict[str, float], List[int]]: Dictionary with the similarity for each graph and 
                the shortest path used to compare the other graph.
        """
        env = gym.make("Maze-v0", **self.params)
        env.load(other_graph)
        other_solution = env.solve(mode="all")
        lengths = [len(sublist) for sublist in other_solution]
        other_solution = other_solution[np.argmin(lengths)]

        similarities = defaultdict(float)
        for i, solution in self.solutions.items():
            similarity = [s1 == s2 for s1, s2 in zip(solution, other_solution)]
            similarities[i] = np.mean(similarity)
        return similarities, other_solution

    def graph_distance(
        self,
        other_graph: Dict[str, Node]
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        """Calculate the distance between the graphs. Distance is calculated as:
            distance = abs(|U| - |I|), where U is the union of the edges and I is the intersection.

        Args:
            other_graph (Dict[str, Node]): Graph to compare.

        Returns:
            Tuple[Dict[str, int], Dict[str, Dict[str, int]]]: Dictionary with the distance for each 
                graph and the distance per level.
        """
        size = int(np.sqrt(max(list(other_graph.keys())) + 1))
        node_level = size * 2 - 1
        number_nodes_level = [x for x in range(1, size+1)]
        number_nodes_level += number_nodes_level[:-1][::-1]
        tree = self.get_tree(node_level, number_nodes_level, size, other_graph)

        distances = defaultdict(int)
        distances_per_level = defaultdict(lambda: defaultdict(int))
        for i, graph in self.graphs.items():
            for level, nodes in tree.items():
                for node in nodes:
                    edges_node = list(graph.values())[node.identifier].edges
                    edges_node = set(edges_node)
                    edges_other = set(node.edges)

                    union = edges_node.union(edges_other)
                    intersection = edges_node.intersection(edges_other)

                    distance = len(union) - len(intersection)
                    distances[i] += distance
                    distances_per_level[i][level] += distance
        return distances, distances_per_level

    def get_tree(
        self,
        node_level: int,
        number_nodes_level: List[int],
        size: int,
        graph: Dict[str, Node]
    ) -> Dict[int, List[Node]]:
        """Get the a tree-like representation of the graph.

        Args:
            node_level (int): How many node level are there in the graph.
            number_nodes_level (List[int]): Number of nodes per level.
            size (int): Size of the maze.
            graph (Dict[str, Node]): Graph to represent.

        Returns:
            Dict[int, List[Node]]: Tree-like representation of the graph.
        """
        tree = {}
        tree[0] = [list(graph.values())[0]]
        for i in range(node_level - 2):
            nodes = list(graph.values())
            if i < node_level // 2:
                level_nodes = [nodes[i+1]]
                if number_nodes_level[i+1] > 2:
                    for n in range(1, number_nodes_level[i+1]-1):
                        index = tree[i][n-1].identifier + size
                        level_nodes += [nodes[index]]
                level_nodes += [nodes[size*(i+1)]]
            else:
                index = tree[i][0].identifier + size
                level_nodes = [nodes[index]]
                if number_nodes_level[i+1] > 2:
                    for n in range(1, number_nodes_level[i+1]-1):
                        index = tree[i][n].identifier + size
                        level_nodes += [nodes[index]]
                index = tree[i][-1].identifier + 1
                level_nodes += [nodes[index]]
            tree[i+1] = level_nodes
        tree[node_level - 1] = [list(graph.values())[-1]]
        return tree

    def draw_node(
        self,
        draw: ImageDraw,
        level_height: float,
        level_width: float,
        message: str
    ) -> Tuple[float, float, float, float]:
        """Draw a node in the image.

        Args:
            draw (ImageDraw): ImageDraw object to draw.
            level_height (float): Height of each level.
            level_width (float): Width of each level.
            message (str): Message to write in the node.

        Returns:
            Tuple[float, float, float, float]: Position of the node. The tuple is in the format of
                (left, top, right, bottom).
        """
        node_size = 60 / 2
        position = (
            level_width - node_size,
            level_height - node_size,
            level_width + node_size,
            level_height + node_size
        )
        draw.ellipse(position, fill="black")
        font = font_manager.FontProperties(family="sans-serif", weight="bold")
        file = font_manager.findfont(font)
        font = ImageFont.truetype(file, 20)
        _, _, w, h = draw.textbbox((0, 0), message, font=font)
        draw.text(
            (level_width - w // 2, level_height - h // 2),
            message,
            font=font
        )
        return position

    def draw_graph(self, graph: Dict[str, Node]) -> Image:
        """Draw the graph in an image.

        Args:
            graph (Dict[str, Node]): Graph to draw.

        Returns:
            Image: PIL Image with the graph.
        """
        size = int(np.sqrt(max(list(graph.keys())) + 1))
        node_level = size * 2 - 1
        edge_level = node_level - 2
        number_nodes_level = [x for x in range(1, size+1)]
        number_nodes_level += number_nodes_level[:-1][::-1]

        tree = self.get_tree(node_level, number_nodes_level, size, graph)

        image = Image.new("RGB", (self.width, self.height), "white")
        draw = ImageDraw.Draw(image)

        node_positions = {}
        levels = node_level + edge_level
        level_height = self.height / (levels + 1)

        # Node
        for level in range(levels+1):
            if level % 2 == 0:
                node_count = number_nodes_level[level//2]
                space_count = node_count + 1
                level_width = self.width / (node_count + space_count)
                for w in range(node_count):
                    _level_height = (level) * level_height
                    _level_height += level_height / 2
                    _level_width = ((w * 2 + 1) * level_width)
                    _level_width += level_width / 2
                    message = str(list(tree.values())[level // 2][w].identifier)
                    positions = self.draw_node(draw, _level_height, _level_width, message)
                    node_positions[int(message)] = positions

        # Edge
        for level in range(levels+1):
            if level % 2 != 0:
                edges_level = (level - 1) // 2
                nodes = list(tree.values())[edges_level]
                for node in nodes:
                    for edge in node.edges:
                        if node.identifier < edge.identifier:
                            source = node_positions[node.identifier][2:]
                            goal = node_positions[edge.identifier][:2]
                        else:
                            source = node_positions[edge.identifier][2:]
                            goal = node_positions[node.identifier][:2]
                        source = (source[0] - 30, source[1])
                        goal = (goal[0] + 30, goal[1])
                        draw.line([goal, source], fill="black", width=2)
        return image

    def collate_graphs(self, graphs: List[Dict[str, Node]]) -> np.ndarray:
        """Collate the graphs in a single image.

        Args:
            graphs (List[Dict[str, Node]]): List of graphs to collate.

        Returns:
            np.ndarray: PIL Image with the graphs.
        """
        image = np.ndarray(shape=(self.height, 0, 3))
        for idx, graph in enumerate(graphs):
            image = np.append(image, np.array(graph), axis=1)
            if idx < len(graphs):
                image = np.append(image, np.zeros((self.height, 2, 3)), axis=1)
        return Image.fromarray(image.astype("uint8"))
