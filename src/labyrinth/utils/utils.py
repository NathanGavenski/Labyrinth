"""Utils package for the labyrinth module."""
from copy import deepcopy
from typing import List, Tuple
import resource
import sys

import numpy as np


class SettingsException(Exception):
    """Exception raised for errors in the settings."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ResetException(Exception):
    """Exception raised when not resetting the labyrinth."""

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
    Find all possile neighbors of a node in a labyrinth like grid.

    Args:
        current_pos : int = node number
        shape : tuple = labyrinth shape (height, width)
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
    Remove redundant nodes for the labyrinth.

    Args:
        edges : list = list of edges (walls) that need 
        to be removed from the labyrinth.

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
    Constructs an array like labyrinth ploting walls and squares.
    Where there is an edge, it removes the wall to create a passage.

    Args:
        edges : list = list of edges (walls) that need to be removed from the labyrinth.
        shape : tuple = labyrinth shape (width, height).
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


def create_mask(
    shape: Tuple[int, int],
    labyrinth: List[List[int]],
    agent: Tuple[int]
) -> List[List[int]]:
    """Create mask for occlusion based on the agent current position and labyrinth structure.

    Args:
        shape (Tuple[int, int]): width and height of the labyrinth
        labyrinth (List[List[int]]): labyrinth structure
        agent (Tuple[int]): (x, y) coordinates for the agent

    Returns:
        List[List[int]]: mask of the given labyrinth for occlusion
    """
    tiles = []
    for height in range(shape[0]):
        for width in range(shape[1]):
            tiles.append(np.array((height, width)))

    labyrinth = labyrinth
    mask = deepcopy(labyrinth)

    for tile in tiles:
        if (tile == agent).all():
            continue
        if (tile == agent).any():
            if tile[1] == agent[1]:  # Vertical mask
                agent_row = agent[0] * 2 + 1
                target_row = tile[0] * 2 + 1
                column = tile[1] * 2 + 1
                lower_bound = agent_row if agent_row < target_row else target_row
                upper_bound = agent_row if agent_row > target_row else target_row
                if (labyrinth[lower_bound:upper_bound + 1, column] == 1).any():
                    mask[target_row, column] = 1
            else:  # Horizontal mask
                agent_column = agent[1] * 2 + 1
                target_column = tile[1] * 2 + 1
                row = tile[0] * 2 + 1
                lower_bound = agent_column if agent_column < target_column else target_column
                upper_bound = agent_column if agent_column > target_column else target_column
                if (labyrinth[row, lower_bound:upper_bound + 1] == 1).any():
                    mask[row, target_column] = 1
        else:  # Diagonal mask
            target_row, target_column = tile * 2 + 1
            agent_row, agent_column = np.array(agent) * 2 + 1

            column_lower_bound = agent_column
            column_upper_bound = target_column
            column = False
            if not agent_column < target_column:
                column_lower_bound = target_column
                column_upper_bound = agent_column
                column = True

            row_lower_bound = agent_row
            row_upper_bound = target_row
            row = False
            if not agent_row < target_row:
                row_lower_bound = target_row
                row_upper_bound = agent_row
                row = True

            matrix = labyrinth[
                row_lower_bound:row_upper_bound + 1,
                column_lower_bound:column_upper_bound + 1
            ]

            if matrix.shape[0] == matrix.shape[1]:
                identity = []
                if row and column:
                    x_range = range(matrix.shape[0] - 1)
                    identity = [[x, x + 1] for x in x_range]
                if not (row and column):
                    identity = [[x + 1, x]
                                for x in range(matrix.shape[0])][:-1]
                if not column and row:
                    x_range = range(matrix.shape[0] - 1, 0, -1)
                    y_range = range(1, matrix.shape[0])
                    identity = [[x, y] for x, y in zip(x_range, y_range)]
                if column and not row:
                    x_range = range(matrix.shape[0] - 1)
                    y_range = range(matrix.shape[0] - 2, -1, -1)
                    identity = [[x, y] for x, y in zip(x_range, y_range)]

                for idx in identity:
                    if matrix[idx[1], idx[0]] == 1:
                        mask[target_row, target_column] = 1
                        continue
            else:
                mask[target_row, target_column] = 1

    return mask
