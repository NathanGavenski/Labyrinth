import random

import numpy as np

class Node:
    def __init__(self, identifier:int, edges:list) -> None:
        self.visited = False
        self.identifier = identifier
        self.visited_edges = []
        
        random.shuffle(edges) # So the iterator is random for the DFS
        self.edges = np.array(edges)
    
    def is_visited(self) -> bool:
        return self.visited

    def visited_from(self, node:int) -> None:
        self.visited = True
        self.visited_edges.append(node)

    def set_neighbor(self, neighbor:list) -> object:
        random.shuffle(neighbor)
        self.edges = np.array(neighbor)
        return self
        
    def get_random_neighbor(self) -> object:
        unvisted_edges = list(
            set(self.edges).difference(self.visited_edges)
        )
        return random.choice(unvisted_edges)
    
    def __getitem__(self, idx:int) -> object:
        return self.edges[idx]

    def __eq__(self, other:object) -> bool:
        return self.identifier == other.identifier

    def __lt__(self, other:object) -> bool:
        return self.identifier < other.identifier

    def __repr__(self) -> str:
        return f'{self.identifier}'

    def __str__(self) -> str:
        return f'{self.identifier}: {self.edges}'
