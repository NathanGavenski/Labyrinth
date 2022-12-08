import ast
from collections import defaultdict
from copy import deepcopy
import os
import random
from typing import List, Tuple

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from .utils import transform_edges_into_walls, Colors
from .utils import get_neighbors, DFS, recursionLimit


class Maze(gym.Env):
    '''
    Description:
        A maze with size [x, y] with an agent (green diamond),
        a start (red square) and a goal (blue square). The goal
        of the objective is for the agent to reach the goal tile.

    Observation:
        Num         Observation         Min     Max
        0           Agent x position    0       shape - 1
        1           Agent y position    0       shape - 1
        2..shape    Maze tile           0       1

        Note: For the rest of the state 0 means an empty 
        space or no wall, and 1 means a wall. Since the 
        maze is a matrix and each tile should represent 
        a current tile, the observation will be (shape * 2).

    Actions:
        Type: Discrete(2)
        Num   Action
        0     UP
        1     RIGHT
        2     DOWN
        3     LEFT

        Note: Walking into a wall will still count as a action.

    Reward:
        Amount          Scenario
        1               For reaching the goal
        -.1 / (x * y)   For all other cases

    Episode Termination:
        Reaching the goal or a fixed number of steps.
    '''

    actions = {
        0: (0, 1),
        1: (1, 0),
        2: (0, -1),
        3: (-1, 0)
    }

    def __init__(
            self,
            shape: tuple,
            start: int = (0, 0),
            end: int = None,
            screen_width: int = 224,
            screen_height: int = 224,
            max_episode_steps: int = 1000,
            occlusion: bool = False,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.viewer = None
        self.state = None
        self.reseted = False
        self.dfs = None
        self.maze = None

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.start = start
        self.end = (self.shape[0] - 1, self.shape[1] - 1) if end is None else end
        self.agent = self.start

        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.occlusion = occlusion

        self.seed()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 255, (screen_width, screen_height, 3), np.uint8)

    def seed(self, seed: int = None) -> List[int]:
        '''
        Set a seed for the environment.
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _generate(self, visited: list = None) -> Tuple[List[Tuple[int]], List[Tuple[int]]]:
        '''
        Create the maze if no maze was loaded.

        Param:
            visited:list = List of edges from a path
        '''
        maze = np.ndarray(shape=self.shape)

        start = (self.start[0] * self.shape[0]) + self.start[1]
        end = self.end[0] * self.shape[0] + self.end[1]

        if visited is None:
            edges = []
            for pos in range(maze.shape[0] * maze.shape[1]):
                edges += get_neighbors(pos, maze.shape, undirected=False)

            self.dfs = DFS(edges, maze.shape, start=start, end=end)
            visited = self.dfs.generate_path([])
        else:
            self.dfs = DFS(visited, maze.shape, start=start, end=end)

        return transform_edges_into_walls(visited, maze.shape), visited

    def get_global_position(self, position: list) -> int:
        '''
        Get global position from a tile.
        '''
        return position[1] * self.shape[0] + position[0]

    def get_local_position(self, position: int) -> List[int]:
        '''
        Get local position from a tile.
        '''
        column = position // self.shape[1]
        row = position - (column * self.shape[1])
        return [column, row]

    def set_occlusion_on(self) -> None:
        '''
        Turn on occlusion mask.
        '''
        self.occlusion = True
    
    def set_occlusion_off(self) -> None:
        '''
        Turn off occlusion mask
        '''
        self.occlusion = False

    def define_pathways(self, pathways):
        _pathways = defaultdict(list)
        for start, end in pathways:
            _pathways[start].append(end)

        self.undirected_pathways = deepcopy(_pathways)
        d = deepcopy(_pathways)
        for key, values in _pathways.items():
            for value in values:
                d[value].append(key)
        return d

    # TODO change to pygame dependency
    def render(self, mode: str = "human"):
        '''
        Render the environment current state.

        Mode:
            Name        description
            human       render a view (image)
            rgb_array   render current state as a numpy array
        '''

        w, h = self.shape
        screen_width = self.screen_width
        screen_height = self.screen_height
        tile_h = screen_height / h
        tile_w = screen_width / w

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

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
            left = self.agent[0] * tile_w
            right = (self.agent[0] + 1) * tile_w
            bottom = self.agent[1] * tile_h
            top = (self.agent[1] + 1) * tile_h
            agent = rendering.FilledPolygon([
                (left + tile_w // 2, bottom),
                (left, top - tile_h // 2),
                (right - tile_w // 2, top),
                (right, bottom + tile_h // 2)
            ])
            self.agent_transition = rendering.Transform()
            agent.add_attr(self.agent_transition)
            agent.set_color(*Colors.GREEN.value)
            self.viewer.add_geom(agent)

            # Draw walls
            for x, tiles in enumerate(self.maze):
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
                            elif x % 2 > 0:  # vertical wall
                                _y = x // 2 + 1
                                _x = y // 2
                                line = rendering.Line(
                                    (_x * tile_w, (_y-1) * tile_h),
                                    (_x * tile_w, _y * tile_h)
                                )
                                line.set_color(*Colors.BLACK.value)
                                self.viewer.add_geom(line)

        # Draw Mask
        if self.occlusion:
            from gym.envs.classic_control import rendering
            mask = self.create_mask()
            for x, tiles in enumerate(mask):
                if (x > 0 and x < self.shape[0] * 2):
                    for y, tile in enumerate(tiles):
                        if tile == 1 and (y > 0 and y < self.shape[0] * 2) and (x % 2 != 0 and y % 2 != 0):
                            _x = x // 2
                            _y = y // 2
                            left = _y * tile_w
                            right = (_y + 1) * tile_w
                            bottom = _x * tile_h
                            top = (_x + 1) * tile_h
                            mask = rendering.FilledPolygon([
                                (left, bottom),
                                (left, top),
                                (right, top),
                                (right, bottom)
                            ])
                            mask.set_color(*Colors.BLACK.value)
                            self.viewer.add_onetime(mask)

        new_x = self.agent[0] * tile_w - self.start[0] * tile_w
        new_y = self.agent[1] * tile_h - self.start[1] * tile_h
        self.agent_transition.set_translation(new_x, new_y)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    # TODO adapt reward function to be 1 - (-.1 / (self.shape[0] * self.shape[1]) * len(shortest_path) )
    def step(self, action: int) -> list:
        '''
        Perform a step in the environment.
        '''
        if action not in self.actions.keys():
            raise Exception(f"Action should be in {self.actions.keys()} it was {action}")

        destiny = np.array(self.agent) + self.actions[action]
        agent_global_position = self.get_global_position(self.agent)
        destiny_global_position = self.get_global_position(destiny)
        if destiny_global_position in self.pathways[agent_global_position]:
            self.agent = tuple(destiny)

        self.step_count += 1
        done = (np.array(self.agent) == self.end).all() or self.step_count >= self.max_episode_steps
        reward = -.1 / (self.shape[0] * self.shape[1]) if not (np.array(self.agent) == self.end).all() else 1

        return self.render('rgb_array'), reward, done, {'state': np.hstack((self.agent, self.maze.flatten()))}

    def reset(self, agent=True, render=True) -> None:
        '''
        Reset the maze. If agent is True, return agent to the start tile.
        If agent is False, return maze to a new one.
        '''
        self.reseted = True
        self.step_count = 0

        if not agent or self.maze is None:
            with recursionLimit(10000):
                self.maze, self._pathways = self._generate()

            self.pathways = self.define_pathways(self._pathways)

        self.agent = self.start
        return self.render('rgb_array') if render else None

    def generate(self, path: str, amount: int = 1) -> None:
        '''
        Create 'n' amount of mazes.
        '''
        for _ in range(amount):
            self.reset(agent=False)
            hash_idx = hash(self)
            if not os.path.exists(path):
                os.makedirs(path)

            file_path = f'{path}/{hash_idx}.txt'
            self.save(file_path)

    def close(self) -> None:
        '''
        Closes the view from the environment.
        Note: This does not reset the environment.
        '''
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def save(self, path: str) -> None:
        '''
        Save the current maze separated by ';'.

        File:
            Position    Description
            0           Maze paths
            1           Start position
            2           Goal position
        '''
        file = path.split('/')[-1]
        path = '/'.join(path.split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f'{path}/{file}', 'w') as f:
            f.write(f'{self._pathways};{self.start};{self.end}')

    def load(self, path: str) -> None:
        '''
        Load the maze from a file.
        '''
        with open(path, 'r') as f:
            for line in f:
                info = line

        visited, start, end = info.split(';')
        pathways = ast.literal_eval(visited)
        self.start = ast.literal_eval(start)
        self.end = ast.literal_eval(end)
        self.maze, self._pathways = self._generate(visited=pathways)
        self.agent = self.start

        self.pathways = self.define_pathways(pathways)
        return self.reset(agent=True, render=False)

    def create_mask(self) -> list:
        '''
        Create mask for occlusion based on the agent current position and maze structure.
        '''
        tiles = []
        for x in range(5):
            for y in range(5):
                tiles.append(np.array((x, y)))
        maze = self.maze
        mask = deepcopy(maze)

        for tile in tiles:
            if (tile == self.agent).all():
                continue
            elif (tile == self.agent).any():
                if tile[1] == self.agent[1]:  # Horizontal mask
                    agent_column = self.agent[0] * 2 + 1
                    target_column = tile[0] * 2 + 1
                    row = tile[1] * 2 + 1
                    lower_bound = agent_column if agent_column < target_column else target_column
                    upper_bound = agent_column if agent_column > target_column else target_column
                    if (maze[row, lower_bound:upper_bound+1] == 1).any():
                        mask[row, target_column] = 1
                else:  # Vertical mask
                    agent_row = self.agent[1] * 2 + 1
                    target_row = tile[1] * 2 + 1
                    column = tile[0] * 2 + 1
                    lower_bound = agent_row if agent_row < target_row else target_row
                    upper_bound = agent_row if agent_row > target_row else target_row
                    if (maze[lower_bound:upper_bound+1, column] == 1).any():
                        mask[target_row, column] = 1
                    pass
            else:  # Diagonal mask FIXME
                target_column, target_row = tile * 2 + 1
                agent_column, agent_row = np.array(self.agent) * 2 + 1

                column_lower_bound, column_upper_bound, column = (agent_column, target_column, False) \
                    if agent_column < target_column \
                    else (target_column, agent_column, True)

                row_lower_bound, row_upper_bound, row = (agent_row, target_row, False) \
                    if agent_row < target_row \
                    else (target_row, agent_row, True)

                matrix = maze[
                    row_lower_bound:row_upper_bound+1, 
                    column_lower_bound:column_upper_bound+1
                ]

                if matrix.shape[0] == matrix.shape[1]:
                    identity = []
                    if row and column:
                        x_range = range(matrix.shape[0] - 1)
                        identity = [[x, x+1] for x in x_range]
                    if not (row and column):
                        identity = [[x + 1, x] for x in range(matrix.shape[0])][:-1]
                    if not column and row:
                        x_range = range(matrix.shape[0]-1, 0, -1)
                        y_range = range(1, matrix.shape[0])
                        identity = [[x, y] for x, y in zip(x_range, y_range)]
                    if column and not row:
                        x_range = range(matrix.shape[0] - 1)
                        y_range = range(matrix.shape[0]-2, -1, -1)
                        identity = [[x, y] for x, y in zip(x_range, y_range)]

                    for idx in identity:
                        if matrix[idx[1], idx[0]] == 1:
                            mask[target_row, target_column] = 1
                            continue
                else:
                    mask[target_row, target_column] = 1

        return mask

    def solve(self, mode: str = 'shortest') -> list:
        '''
        Solve the current maze

        Param:
            mode = amount of paths to return [shortest/all].
        '''
        with recursionLimit(100000):
            paths = self.dfs.find_paths(self.pathways)

        if mode == 'shortest':
            return min(paths)
        else:
            return paths

    def change_start_and_goal(self, min_distance=None) -> tuple:
        '''
        '''
        if min_distance is None:
            w, h = self.shape
            min_distance = (w + h) // 2

        paths = self.solve(mode='all')
        longest_path = np.argmax(max(len(path) for path in paths))
        path = paths[longest_path]
        start = np.array(
            self.get_local_position(
                np.random.choice(path[1:], 1)[0]
            )
        )

        possible_goals = []
        for tile in path:
            tile = self.get_local_position(tile)
            distance = np.abs(start - tile).sum()
            if distance >= min_distance:
                possible_goals.append(tile)
        
        if len(possible_goals) == 0:
            return self.change_start_and_goal(min_distance)
        else:
            end_idx = np.random.choice(
                [x for x in range(len(possible_goals))], 
                1
            )[0]
            end = possible_goals[end_idx]
            self.start = tuple(start)
            self.end = tuple(end)
            self.agent = self.start

            self.dfs.start = self.get_global_position(self.start)
            self.dfs.end = self.get_global_position(self.end)
            return start, end

    def agent_random_position(self) -> None:
        self.reset()
        self.agent = (
            random.randint(0, self.shape[0] - 1),
            random.randint(0, self.shape[1] - 1)
        )

    def __hash__(self) -> int:
        '''
        Create a hash of the edges of the maze.

        Note: in order to have a consistent hash, 
        it sorts the inner tuples (the edges), and
        the tuples (the list of edges), and remove 
        duplicates before hashing the maze.
        '''
        pathways = list(map(sorted, self._pathways))
        pathways.sort()
        pathways = tuple(set(map(tuple, pathways)))
        return hash(pathways)
