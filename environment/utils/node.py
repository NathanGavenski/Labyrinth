import random

import numpy as np

class Node:
    def __init__(self, identifier, edges):
        self.visited = False
        self.identifier = identifier
        random.shuffle(edges)
        self.edges = np.array(edges)
    
    def is_visited(self):
        return self.visited

    def visited_from(self, node):
        self.visited = True
        self.edges = np.delete(
            self.edges, 
            np.where(self.edges == node)
        )
        
    def get_random_neighbor(self):
        return random.choice(self.edges)
    
    def __getitem__(self, idx):
        return self.edges[idx]
