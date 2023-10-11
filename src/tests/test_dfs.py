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

        self.dfs = DFS(self.edges, (3, 3))

    def tearDown(self) -> None:
        pass

    def test_init(self) -> None:
        assert self.dfs.start == 0
        assert self.dfs.end == 8
        assert not self.dfs.key_and_door
        assert self.dfs.update is None
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

    def test_are_all_subsets(self) -> None:
        assert self.dfs.are_all_subsets(
            [1, 2, 3],
            [[4, 5, 6, 8], [1, 2, 3, 8]],
            Node(8, [])
        )
        assert self.dfs.are_all_subsets(
            [1, 2, 3, 8],
            [[4, 5, 6, 8], [1, 2, 5, 8]],
            Node(8, [])
        )
        assert not self.dfs.are_all_subsets(
            [1, 2, 4],
            [[4, 5, 6, 8], [1, 2, 3, 8]],
            Node(8, [])
        )

    def test_are_all_lists_subsets(self) -> None:
        assert self.dfs.are_all_lists_subsets(
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3, 8], [4, 5, 6, 8]],
            Node(8, [])
        )
        assert not self.dfs.are_all_lists_subsets(
            [[1, 2, 3], [4, 7, 6]],
            [[1, 2, 3, 8], [4, 5, 6, 8]],
            Node(8, [])
        )

    def test_find_loop(self) -> None:
        self.dfs.convert_graph()
        assert not self.dfs.find_loop(self.dfs.graph)

        self.dfs.graph[0].edges[0].add_edge(self.dfs.graph[0])
        assert self.dfs.find_loop(self.dfs.graph)
