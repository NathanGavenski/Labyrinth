from collections import defaultdict

from .node import Node
from .utils import recursionLimit

class DFS:
    def __init__(self, graph: list, shape: tuple, start : int = None, end : int = None):
        '''
        Depth first search algorithm for the maze generation.

        graph : list = list of edges for each cell (ex: [(0, 1), (0, 5)])
        shape : tuple = size of the map (width, height)
        start : int = absolute position for the start (default: 0)
        end : int = absolute position for the end (default: the last cell - up left most cell)
        '''
        self.start = 0 if start is None else start
        self.end = shape[0] * shape[1] - 1 if end is None else end
        self.shape = shape
        self.graph = graph
        self.nodes = []
        self.paths = []
        self.reset()
        
    def reset(self):
        '''
        Reset the Node graph.
        '''
        del self.nodes

        edges_dict = defaultdict(list)
        for key, neighbor in self.graph:
            edges_dict[key].append(neighbor)
    
        self.nodes = []
        for node in range(self.shape[0] * self.shape[1]):
            self.nodes.append(Node(node, []))

        for key, value in edges_dict.items():
            self.nodes[key].set_neighbor(value)
            
    def generate_path(self, visited:list, start:int=None) -> list:
        '''
        Generates a maze-like with DFS. 

        visited: list of visited nodes (default: empty)
        start: where to start the maze (default: None - self.start)
        
        Returns a list of tuples with all edges that form the paths.
        To form a maze remove these edges from the env.
        '''
        current = self.start if start is None else start
        
        if start is None:
            self.nodes[current].visited = True
        
        if current == self.end:
            self.nodes[current].visited = False
        
        for node in self.nodes[current]:
            node = self.nodes[node]
            if not node.is_visited():
                node.visited_from(current)
                visited.append((current, node.identifier))
                visited = self.generate_path(visited, node.identifier)
        return visited

    def find_paths(self, edges: defaultdict) -> list:
        '''
        Discover all possible paths to the goal of the maze.
        edges: defaultdict with the node identifier and its neighbors.

        Returns a list of list of all the nodes that take the agent to its goal.
        '''
        nodes_dict = {x: Node(x, []) for x in edges.keys()}

        for x, y in edges.items():
            for node in y:
                nodes_dict[x].add_edge(nodes_dict[node])
            
        self.path = []
        self._find_paths(set(), nodes_dict, nodes_dict[self.start], [])
        return self.path
            
    def _find_paths(self, visited, graph, node, path) -> None:
        '''
        Auxiliary recursion function for the find_paths().
        '''
        path = list(tuple(path))

        if node.identifier == self.end:
            path.append(node.identifier)
            self.path.append(path)
            return

        if node.identifier not in visited:
            path.append(node.identifier)
            visited.add(node.identifier)
            for neighbor in node:
                self._find_paths(visited, graph, neighbor, path)
