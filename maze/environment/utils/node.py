import random

import numpy as np

class Node:
    def __init__(self, identifier:int, edges:list) -> None:
        self.visited = False
        self.identifier = identifier
        self.visited_edges = []
        
        random.shuffle(edges) # So the iterator is random for the DFS
        self.edges = edges
    
    def is_visited(self) -> bool:
        return self.visited

    def add_edge(self, edge: object) -> None:
        self.edges.append(edge)

    def visited_from(self, node: object) -> None:
        self.visited = True
        if isinstance(node, int):
            self.visited_edges.append(node)
        else:
            self.visited_edges.append(node.identifier)

    def set_neighbor(self, neighbor:list) -> object:
        random.shuffle(neighbor)
        self.edges = np.array(neighbor)
        return self
        
    def get_random_neighbor(self) -> object:
        unvisted_edges = list(
            set(self.edges).difference(self.visited_edges)
        )
        return random.choice(unvisted_edges)
    
    def __len__(self):
        current_available_edges = []
        for node in self.edges:
            if not node.visited:
                current_available_edges.append(node)
        return len(current_available_edges)

    def __getitem__(self, idx:int) -> object:
        return self.edges[idx]

    def __eq__(self, other:object) -> bool:
        return self.identifier == other.identifier

    def __lt__(self, other:object) -> bool:
        return self.identifier < other.identifier

    def __hash__(self) -> int:
        return self.identifier

    def __repr__(self) -> str:

        return f'{[edge.identifier for edge in self.edges]}'

    def __str__(self) -> str:
        current_available_edges = []
        for node in self.edges:
            if not node.visited:
                current_available_edges.append(node)

        return f'{self.identifier}: {[edge.identifier for edge in current_available_edges]}'
