from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def get_neighbors(current_pos : int, shape : tuple) -> list:
    '''

    '''
    neighbors = np.array([-shape[1], -1, +shape[1], +1]) # up, right, down, left
    possible_neighbors = np.array(current_pos - neighbors)
    delete = []

    # Test up neighbor
    if possible_neighbors[0] > shape[0] * shape[1] - shape[1]:
        delete.append(0)

    # Test right neighbor
    if possible_neighbors[1] % shape[1] == 0:
        delete.append(1)

    # Test down neighbor
    if possible_neighbors[2] < 0:
        delete.append(2)

    # Test left neighbor
    left = int(possible_neighbors[3] / shape[1])
    pos = int(current_pos / shape[1])
    if left != pos or possible_neighbors[3] < 0:
        delete.append(3)

    neighbors = []
    for neighbor in np.delete(possible_neighbors, delete, 0):
        neighbors.append((current_pos, neighbor))
    return neighbors

def plot_graph(edges, shape):
    '''
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