import gym
from gym.utils import seeding
import numpy as np

from utils import get_neighbors, DFS, transform_edges_into_walls


class Maze(gym.Env):
    def __init__(self, shape:tuple) -> None:
        super().__init__()
        self.shape = shape
        self.viewer = None
        self.state = None
        self.seed()
    
    def seed(self, seed=None) -> list:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate(self) -> list:
        maze = np.ndarray(shape=self.shape)

        edges = []
        for pos in range(maze.shape[0] * maze.shape[1]):
            edges += get_neighbors(pos, maze.shape, undirected=False)

        dfs = DFS(edges, maze.shape)
        visited = dfs.generate_path([])
        return transform_edges_into_walls(visited, maze.shape)

    def render(self, mode:str = "human"):
        raise NotImplementedError('')

    def step(self, action:int) -> list:
        raise NotImplementedError('')

    def close(self) -> None:
        raise NotImplementedError('')

    def save(self, path:str) -> None:
        raise NotImplementedError('')
    
    def load(self, path:str) -> None:
        raise NotImplementedError('')
