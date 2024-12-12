"""Module for testing DFS class."""
from collections import defaultdict
from unittest import TestCase

import pytest

from src.maze.utils.dfs import DFS
from src.maze.utils.node import Node


class TestDFS(TestCase):

    def setUp(self) -> None:
        self.graph = {
            0: [1, 3],
            1: [2, 4],
            2: [5],
            3: [4, 6],
            4: [5, 7],
            5: [8],
            6: [7],
            7: [8],
            8: []
        }

        self.edges = defaultdict(list)
        for key, value in self.graph.items():
            for edge in value:
                self.edges[key].append((key, edge))
        self.edges[8].append(())

        self.dfs = DFS(self.edges, (3, 3))

    def tearDown(self) -> None:
        pass

    def test_init(self) -> None:
        assert self.dfs.start == 0
        assert self.dfs.end == 8
        assert not self.dfs.key_and_door
        assert self.dfs.random_amount == 0

    def test_convert_graph(self) -> None:
        self.dfs.convert_graph()
        for o_dict, t_dict in zip(self.graph.items(), self.dfs.graph.items()):
            o_key, o_val = o_dict
            t_key, t_val = t_dict
            t_val.edges.sort()
            assert o_val == t_val.edges

    def test_set_key_and_door(self) -> None:
        with pytest.raises(ValueError):
            self.dfs.set_key_and_door((1, 1), 2)

        with pytest.raises(ValueError):
            self.dfs.set_key_and_door(2, (1, 1))

        self.dfs.set_key_and_door(2, 3)
        assert self.dfs.key == 2
        assert self.dfs.door == 3
        assert self.dfs.key_and_door

    def test_generate_path(self) -> None:
        graph = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7]
        }
        edges = defaultdict(list)
        for key, value in graph.items():
            for edge in value:
                edges[key].append((key, edge))

        dfs = DFS(edges, (3, 3))
        maze = dfs.generate_path()

        # Check if the maze is a dictionary with Node objects
        assert isinstance(maze, dict)
        assert isinstance(maze[0], Node)

        # Check if some walls have been removed
        count = 0
        for key, value in maze.items():
            if value.edges != graph[key]:
                count += 1
        assert count > 0

        dfs = DFS(edges, (3, 3), random_amount=25)
        maze = dfs.generate_path(min_paths=2)
        solutions = dfs.find_paths(maze, False)
        assert len(solutions) >= 2

        dfs = DFS(edges, (3, 3), random_amount=25)
        maze = dfs.generate_path(max_paths=2)
        solutions = dfs.find_paths(maze, False)
        assert len(solutions) <= 2

    def test__generate_path(self) -> None:
        _graph = {
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7]
        }
        graph = {x: Node(x, []) for x in self.graph}
        for key, value in _graph.items():
            for edge in value:
                graph[key].add_edge(graph[edge])

        self.random_amount = 0
        self.dfs._generate_path(graph[0], graph[0], graph[8])

        # Check if some walls have been removed
        count = 0
        for node in graph.values():
            if node.edges != _graph[node.identifier]:
                count += 1
        assert count > 0

    def test_find_paths(self) -> None:
        graph = {x: Node(x, []) for x in self.graph}
        for key, value in self.graph.items():
            for edge in value:
                graph[key].add_edge(graph[edge])

        path = self.dfs.find_paths(graph, True)
        assert isinstance(path, list)
        assert isinstance(path[0], list)
        assert isinstance(path[0][0], Node)
        assert len(path) == 1

        path = self.dfs.find_paths(graph, False)
        assert isinstance(path, list)
        assert isinstance(path[0], list)
        assert isinstance(path[0][0], Node)
        assert len(path) == 6


    def test_find_path(self) -> None:
        graph = {x: Node(x, []) for x in self.graph}
        for key, value in self.graph.items():
            for edge in value:
                graph[key].add_edge(graph[edge])

        path = self.dfs.find_path(graph, 0, 8, False)
        assert isinstance(path, dict)
        assert isinstance(path[8], Node)

    def test__find_path(self) -> None:
        graph = {x: Node(x, []) for x in self.graph}
        for key, value in self.graph.items():
            for edge in value:
                graph[key].add_edge(graph[edge])

        self.dfs._find_path(graph[0].d, graph[0], graph[8], False)

        for key, value in graph.items():
            for node in value:
                assert key in graph[node].visited_edges

    def test__find_all_paths(self) -> None:
        graph = {x: Node(x, []) for x in self.graph}
        for key, value in self.graph.items():
            for edge in value:
                graph[key].add_edge(graph[edge])

        visited = [False] * len(self.graph)
        self.dfs._find_all_paths(graph[0], graph[8], visited, [])

        solutions = [
            [0, 1, 2, 5], [0, 1, 4, 7], [0, 1, 4, 5],
            [0, 3, 4, 7], [0, 3, 4, 5], [0, 3, 6, 7]
        ]

        d = sorted(sorted(sublist) for sublist in graph[8].d)
        solutions = sorted(sorted(sublist) for sublist in solutions)
        assert solutions == d
 
