from collections import defaultdict

from .node import Node

class DFS:
    def __init__(self, graph, shape, start=None, end=None):
        self.start = 0 if start is None else start
        self.end = shape[0] * shape[1] - 1 if end is None else end
        self.shape = shape
        self.graph = graph
        self.nodes = []
        self.reset()
        
    def reset(self):
        del self.nodes

        edges_dict = defaultdict(list)
        for key, neighbor in self.graph:
            edges_dict[key].append(neighbor)
    
        self.nodes = []
        for node in range(self.shape[0] * self.shape[1]):
            self.nodes.append(Node(node, []))

        for key, value in edges_dict.items():
            self.nodes[key].set_neighbor(value)
            
    def generate_path(self, visited, start=None):
        current = self.start if start is None else start
        
        if start is None:
            self.nodes[current].visited = True
        
        if current == self.end:
            return visited
        
        for node in self.nodes[current]:
            node = self.nodes[node]
            if not node.is_visited():
                node.visited_from(current)
                visited.append((current, node.identifier))
                visited = self.generate_path(visited, node.identifier)
        return visited
