"""Utils package for the maze module."""
import resource
import sys

import numpy as np


class SettingsException(Exception):
    """Exception raised for errors in the settings."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ResetException(Exception):
    """Exception raised when not resetting the maze."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ActionException(Exception):
    """Action not in action space."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


def get_neighbors(current_pos: int, shape: tuple, undirected: bool = False) -> list:
    '''
    Find all possile neighbors of a node in a maze like grid.

    Args:
        current_pos : int = node number
        shape : tuple = maze shape (height, width)
        undirected : bool = if the direction of the nodes is relevant

    For example:
        Directed nodes: [0, 1] and [1, 0] are viable options
        Undirected nodes: only [0, 1] is a viable option
    '''
    width, height = shape
    if not undirected:
        neighbors = np.array([+width, +1, -width, -1])  # up, right, down, left
    else:
        neighbors = np.array([+width, +1])  # up, right

    possible_neighbors = np.array(current_pos + neighbors)

    delete = []

    # Test up neighbor
    if possible_neighbors[0] > width * height - 1:
        delete.append(0)

    # Test right neighbor
    if possible_neighbors[1] % width == 0:
        delete.append(1)

    # Test down neighbor
    if not undirected and possible_neighbors[2] < 0:
        delete.append(2)

    # Test left neighbor
    if not undirected:
        left = int(possible_neighbors[3] / width)
        pos = int(current_pos / width)
        if left != pos or possible_neighbors[3] < 0:
            delete.append(3)

    neighbors = []
    for neighbor in np.delete(possible_neighbors, delete, 0):
        neighbors.append((current_pos, neighbor))

    return neighbors


def remove_redundant_nodes(edges: list) -> list:
    '''
    Remove redundant nodes for the maze.

    Args:
        edges : list = list of edges (walls) that need 
        to be removed from the maze.

    Return:
        It returns a list of tuples (list and array are not
        hashables, so this allows for set functions - difference)

    For example:
        (0, 1) is the same as (1, 0)
    '''
    tobe_deleted = []
    for idx, edge in enumerate(edges):
        idx = idx + 1
        for _idx, _edge in enumerate(edges[idx:]):
            if set(edge) == set(_edge):
                tobe_deleted.append(idx+_idx)
    return list(map(tuple, np.delete(edges, tobe_deleted, 0)))


def transform_edges_into_walls(edges: list, shape: tuple) -> list:
    """
    Constructs an array like maze ploting walls and squares.
    Where there is an edge, it removes the wall to create a passage.

    Args:
        edges : list = list of edges (walls) that need to be removed from the maze.
        shape : tuple = maze shape (width, height).
    """
    width, height = shape
    walls = np.zeros(shape=[height * 2 + 1, width * 2 + 1])
    vertical_walls = list(range(0, height * 2 + 1, 2))
    horizontal_walls = list(range(0, width * 2 + 1, 2))
    walls[:, vertical_walls] = 1
    walls[horizontal_walls, :] = 1
    for edge in edges:
        if int(edge[0] / width) == int(edge[1] / width):  # same row
            x_position, y_position = edge
            _max, _min = max(x_position, y_position), min(x_position, y_position)
            row = int(_min / width) * 2 + 1
            column = vertical_walls[_max - (int(_max/width) * width)]
            walls[row, column] = 0
        else:  # different row
            x_position, y_position = edge
            _max, _min = max(x_position, y_position), min(x_position, y_position)
            row = horizontal_walls[int(_min/width)+1]
            column = (_max - int(_max/width) * width) * 2 + 1
            walls[row, column] = 0
    return walls


class RecursionLimit:
    """Class for setting a higher recursion limit."""

    def __init__(self, limit: int) -> None:
        """Set the recursion limit.

        Args:
            limit (int): Recursion limit.
        """
        self.limit = limit
        self.old_memory_limit = None
        self.old_size_limit = None

    def __enter__(self) -> None:
        """Enter the recursion limit context."""
        self.old_size_limit = sys.getrecursionlimit()
        self.old_memory_limit = resource.getrlimit(resource.RLIMIT_STACK)
        sys.setrecursionlimit(self.limit)
        resource.setrlimit(
            resource.RLIMIT_STACK,
            (0x10000000, resource.RLIM_INFINITY)
        )

    def __exit__(self, *args) -> None:
        """Exit the recursion limit context."""
        sys.setrecursionlimit(self.old_size_limit)
        resource.setrlimit(
            resource.RLIMIT_STACK,
            self.old_memory_limit
        )
