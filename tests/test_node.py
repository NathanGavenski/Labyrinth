from unittest import TestCase

import pytest

from src.labyrinth.utils.node import Node


class TestNode(TestCase):
    """Test cases for testing Node class."""

    def setUp(self) -> None:
        self.node = Node(0, [])
        edge = Node(1, [])

        self.node.add_edge(edge)
        edge.visited_from(self.node)

    def test_add_edge(self) -> None:
        assert self.node.edges[0] == 1
        assert len(self.node.edges) == 1

        self.node.add_edge(Node(1, []))
        assert len(self.node.edges) == 1

        self.node.add_edge(Node(2, []))
        assert len(self.node.edges) == 2

    def test_remove_parent(self) -> None:
        assert len(self.node.edges) == 1
        assert len(self.node.visited_edges) == 0

        self.node.remove_parent()
        assert len(self.node.edges) == 1
        assert len(self.node.directed_edges) == 1

        self.node.visited_from(Node(1, []))
        assert len(self.node.visited_edges) == 1

        self.node.remove_parent()
        assert len(self.node.edges) == 0
        assert len(self.node.directed_edges) == 0
        assert len(self.node.visited_edges) == 0

    def test_remove_edge(self) -> None:
        node = Node(1, [])
        node.visited_from(self.node)
        assert self.node.edges[0] == node

        self.node.remove_edge(node)
        assert node not in self.node.edges
        assert node in self.node.walls
        with pytest.raises(ValueError):
            self.node.remove_edge(node)

        self.node.walls = []
        self.node.add_edge(node)
        self.node.remove_edge_no_walls(node)
        assert node not in self.node.walls
        assert node not in self.node.edges
        assert self.node.remove_edge_no_walls(node)

    def test_add_d(self) -> None:
        assert len(self.node.d) == 0

        d = [self.node, Node(1, []), Node(2, [])]
        self.node.add_d(d)
        assert len(self.node.d) == 1
        assert isinstance(self.node.d[0], list)

        self.node.add_d(d)
        assert len(self.node.d) == 1

        self.node.add_d(d[1:])
        assert len(self.node.d) == 2

    def test_compare(self) -> None:
        node = Node(0, [])
        assert node == self.node
        assert 0 == self.node

        node.identifier = 1
        assert not node == self.node
        assert not 1 == self.node

        assert node > self.node
        assert 1 > self.node

        assert not node < self.node
        assert not 0 < self.node
