import resource
import sys

import matplotlib.pyplot as plt
import numpy as np


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
    w, h = shape
    if not undirected:
        neighbors = np.array([+w, +1, -w, -1])  # up, right, down, left
    else:
        neighbors = np.array([+w, +1])  # up, right

    possible_neighbors = np.array(current_pos + neighbors)
    delete = []

    # Test up neighbor
    if possible_neighbors[0] > w * h - 1:
        delete.append(0)

    # Test right neighbor
    if possible_neighbors[1] % w == 0:
        delete.append(1)

    # Test down neighbor
    if not undirected and possible_neighbors[2] < 0:
        delete.append(2)

    # Test left neighbor
    if not undirected:
        left = int(possible_neighbors[3] / w)
        pos = int(current_pos / w)
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
    '''
    Constructs an array like maze ploting walls and squares.
    Where there is an edge, it removes the wall to create a passage.

    Args:
        edges : list = list of edges (walls) that need 
        to be removed from the maze.
        shape : tuple = maze shape (width, height).
    '''
    w, h = shape
    walls = np.zeros(shape=[h*2+1, w*2+1])
    vertical_walls = list(range(0, h*2+1, 2))
    horizontal_walls = list(range(0, w*2+1, 2))
    walls[:, vertical_walls] = 1
    walls[horizontal_walls, :] = 1
    for edge in edges:
        if int(edge[0] / w) == int(edge[1] / w):  # same row
            x, y = edge
            _max, _min = max(x, y), min(x, y)
            row = int(_min / w) * 2 + 1
            column = vertical_walls[_max - (int(_max/w) * w)]
            walls[row, column] = 0
        else:  # different row
            x, y = edge
            _max, _min = max(x, y), min(x, y)
            row = horizontal_walls[int(_min/w)+1]
            column = (_max - int(_max/w) * w) * 2 + 1
            walls[row, column] = 0
    return walls


def plot_graph(edges: list, shape: tuple) -> None:
    '''
    Plot the edges in a graph format.

    Args:
        edges : list = list of edges (walls) that need 
        to be removed from the maze.
        shape : tuple = maze shape (height, width).
    '''
    edges = sorted(edges)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    pos = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            pos.append((x, y))
    nx.drawing.nx_pylab.draw_networkx(graph, pos)
    plt.show()


class recursionLimit:
    '''
    Class for setting a higher recursion limit.
    '''

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def __enter__(self) -> None:
        self.old_size_limit = sys.getrecursionlimit()
        self.old_memory_limit = resource.getrlimit(resource.RLIMIT_STACK)
        sys.setrecursionlimit(self.limit)
        resource.setrlimit(
            resource.RLIMIT_STACK,
            (0x10000000, resource.RLIM_INFINITY)
        )

    def __exit__(self, *args) -> None:
        sys.setrecursionlimit(self.old_size_limit)
        resource.setrlimit(
            resource.RLIMIT_STACK,
            self.old_memory_limit
        )
