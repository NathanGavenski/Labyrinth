import random

import numpy as np

class Node:
    def __init__(self, identifier, edges):
        self.visited = False
        self.identifier = identifier
        self.visited_edges = []
        
        random.shuffle(edges) # So the iterator is random for the DFS
        self.edges = np.array(edges)
    
    def is_visited(self):
        return self.visited

    def visited_from(self, node):
        self.visited = True
        self.visited_edges.append(node)

    def set_neighbor(self, neighbor):
        random.shuffle(neighbor)
        self.edges = np.array(neighbor)
        
    def get_random_neighbor(self):
        unvisted_edges = list(
            set(self.edges).difference(self.visited_edges)
        )
        return random.choice(unvisted_edges)
    
    def __getitem__(self, idx):
        return self.edges[idx]
