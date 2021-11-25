import os
import ast

import gym
from gym.utils import seeding
import numpy as np

from utils import transform_edges_into_walls, Colors
from utils import get_neighbors, DFS, recursionLimit


class Maze(gym.Env):
    def __init__(self, shape : tuple, start : int = (0, 0), end : int = None) -> None:
        super().__init__()
        self.shape = shape
        self.viewer = None
        self.state = None
        self.reseted = False
        
        self.start = start 
        self.end = (self.shape[0] - 1, self.shape[1] - 1) if end is None else end

        self.seed()
    
    def seed(self, seed=None) -> list:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate(self, visited : list = None) -> list:
        maze = np.ndarray(shape=self.shape)

        if visited is None:
            edges = []
            for pos in range(maze.shape[0] * maze.shape[1]):
                edges += get_neighbors(pos, maze.shape, undirected=True)

            start = (self.start[0] * self.shape[0]) + self.start[1]
            end = (self.end[0] * self.shape[0]) + self.end[1]

            dfs = DFS(edges, maze.shape, start=start, end=end)
            visited = dfs.generate_path([])

        return transform_edges_into_walls(visited, maze.shape), visited

    def render(self, mode : str = "human"):
        if not self.reseted:
            raise Exception('You should reset first.')

        w, h = self.shape
        screen_width = 600
        screen_height = 600

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            tile_h = screen_height // h
            tile_w = screen_width // w

            for x, tiles in enumerate(self.maze[::-1]):
                if (x > 0 and x < self.shape[0] * 2):
                    for y, tile in enumerate(tiles):
                        if tile == 1 and (y > 0 and y < self.shape[0] * 2):
                            if x % 2 == 0 and (y % 2 != 0 or y == 1):  # horizontal wall
                                _y = x // 2
                                _x = y // 2 + 1
                                line = rendering.Line(
                                    ((_x-1) * tile_w, _y * tile_h),
                                    (_x * tile_w, _y * tile_h)
                                )
                                line.set_color(*Colors.BLACK.value)
                                self.viewer.add_geom(line)   
                            elif x % 2 > 0: # vertical wall
                                _y = x // 2 + 1
                                _x = y // 2
                                line = rendering.Line(
                                    (_x * tile_w, (_y-1) * tile_h),
                                    (_x * tile_w, _y * tile_h)
                                )
                                line.set_color(*Colors.BLACK.value)
                                self.viewer.add_geom(line)                                 

            # Draw start
            left = self.start[0] * tile_w
            right = (self.start[0] + 1) * tile_w
            top = self.start[1] * tile_h
            bottom = (self.start[1] + 1) * tile_h
            start = rendering.FilledPolygon([
                (left, bottom),
                (left, top),
                (right, top),
                (right, bottom)
            ])
            start.set_color(*Colors.RED.value)
            self.viewer.add_geom(start)

            # Draw end
            left = self.end[0] * tile_w
            right = (self.end[0] + 1) * tile_w
            top = self.end[1] * tile_h
            bottom = (self.end[1] + 1) * tile_h
            end = rendering.FilledPolygon([
                (left, bottom),
                (left, top),
                (right, top),
                (right, bottom)
            ])
            end.set_color(*Colors.BLUE.value)
            self.viewer.add_geom(end)

            # Draw agent
            agent_pos = self.start
            left = agent_pos[0] * tile_w
            right = (agent_pos[0] + 1) * tile_w
            bottom = agent_pos[1] * tile_h
            top = (agent_pos[1] + 1) * tile_h
            agent = rendering.FilledPolygon([
                (left + tile_w // 2, bottom),
                (left, top - tile_h // 2),
                (right - tile_w // 2, top),
                (right, bottom + tile_h // 2)
            ])
            agent.set_color(*Colors.GREEN.value)
            self.viewer.add_geom(agent)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")    

    def step(self, action:int) -> list:
        raise NotImplementedError('')

    def reset(self) -> None:
        self.reseted = True

        with recursionLimit(10000):
            self.maze, self.pathways = self.generate()

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def save(self, path:str) -> None:
        file = path.split('/')[-1]
        path = '/'.join(path.split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f'{path}/{file}', 'w') as f:
            f.write(f'{self.pathways};{self.start};{self.end}')
    
    def load(self, path:str) -> None:
        with open(path, 'r') as f:
            for line in f:
                info = line

        visited, start, end = info.split(';')
        pathways = ast.literal_eval(visited)
        self.start = ast.literal_eval(start)
        self.end = ast.literal_eval(end)
        self.maze, self.pathways = self.generate(visited=pathways)
        self.reseted = True

if __name__ == '__main__':
    from PIL import Image
    
    maze = Maze((5, 5))
    maze.reset()
    Image.fromarray(maze.render('rgb_array')).save('tst01.png')
    maze.save('./maze/01.txt')
    maze.close()
    del maze

    maze = Maze((5, 5))
    maze.load('./maze/01.txt')
    Image.fromarray(maze.render('rgb_array')).save('tst02.png')
    maze.close()