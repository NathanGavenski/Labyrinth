"""Maze environment."""
import ast
from collections import defaultdict
from copy import deepcopy
import os
import random
from typing import List, Tuple, Optional, Union

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from numpy import ndarray

from .utils import transform_edges_into_walls
from .utils import get_neighbors, DFS, RecursionLimit
from .utils import SettingsException, ResetException, ActionException
from .utils.render import RenderUtils


# FIXME typing
class Maze(gym.Env):
    """
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
    """

    actions = {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1)
    }

    def __init__(
            self,
            shape: Tuple[int, int],
            start: Tuple[int, int] = (0, 0),
            end: Optional[Tuple[int, int]] = None,
            screen_width: Optional[int] = 224,
            screen_height: Optional[int] = 224,
            max_episode_steps: Optional[int] = 1000,
            occlusion: Optional[bool] = False,
            key_and_door: Optional[bool] = False,
            icy_floor: Optional[bool] = False,
    ) -> None:
        """Maze environment.

        Args:
            shape (Tuple[int, int]): shape of the maze (width, height).
            start (Tuple[int, int], optional): Start tile of the maze. Defaults to (0, 0).
            end (Tuple[int, int], optional): End tile for the maze. Defaults to None.
            screen_width (int, optional): Width for image size (pixels). Defaults to 224.
            screen_height (int, optional): Height for image size (pixels). Defaults to 224.
            max_episode_steps (int, optional): Max number of steps per episode. Defaults to 1000.
            occlusion (bool, optional): Whether occlusion setting should be use. Defaults to False.
            key_and_door (bool, optional): Whether key and door setting should be use. 
                Defaults to False.
            icy_floor (bool, optional): Whether icy floor seeting should be use. Defaults to False.

        Raises:
            SettingsException: Occlusion and key and door cannot be active at the same time.
        """
        super().__init__()
        self.shape = shape
        self.render_utils = None
        self.state = None
        self.reseted = False
        self.dfs = None
        self.maze = None
        self.undirected_pathways = None
        self.agent_transition = None
        self.pathways = None
        self._pathways = None

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.start = start
        self.end = (self.shape[0] - 1, self.shape[1] -
                    1) if end is None else end
        self.agent = self.start

        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.occlusion = occlusion

        self.seed()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            0, 255, (screen_width, screen_height, 3), np.uint8)

        self.key_and_door = key_and_door
        self.key, self.door = None, None

        self.icy_floor = icy_floor
        self.ice_floors = None

        if self.key_and_door and self.occlusion:
            raise SettingsException(
                "Both modes cannot be active at the same time.")

    def seed(self, seed: int = None) -> List[int]:
        """
        Set a seed for the environment.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _generate(
            self,
            visited: List[int] = None,
            initial: List[int] = None,
            goal: List[int] = None,
            min_paths: int = None,
    ) -> Tuple[List[Tuple[int]], List[Tuple[int]]]:
        """Create a maze using DFS algorithm.

        Args:
            visited (List[int], optional): List of visited nodes. Defaults to None.

        Returns:
            Tuple[List[Tuple[int]], List[Tuple[int]]]: Maze and visited nodes.
        """
        maze = np.ndarray(shape=self.shape)

        initial = self.start if initial is None else initial
        goal = self.end if goal is None else goal

        start = (initial[0] * self.shape[0]) + initial[1]
        end = goal[0] * self.shape[0] + goal[1]

        if visited is None:
            edges = {}
            for pos in range(maze.shape[0] * maze.shape[1]):
                edges[pos] = get_neighbors(pos, maze.shape, undirected=False)

            self.dfs = DFS(edges, maze.shape, start=start, end=end)
            with RecursionLimit(100000):
                graph = self.dfs.generate_path(min_paths=min_paths)

            visited = []
            for node in graph.values():
                for edge in node.directed_edges:
                    visited.append((node.identifier, edge.identifier))
        else:
            edges = defaultdict(list)
            for (x, y) in visited:
                edges[x].append((x, y))
            self.dfs = DFS(edges, maze.shape, start=start, end=end)
            self.dfs.convert_graph()

        return transform_edges_into_walls(visited, maze.shape), visited

    def get_global_position(
            self,
            position: Union[List[int], Tuple[int, int]],
            size: Tuple[int, int] = None
    ) -> int:
        """Get global position from a tile.

        Args:
            position (List[int] | Tuple[int, int]): (x, y) coordinates
            size (List[int], optional): Size of the maze. Defaults to None.

        Returns:
            int: Global position.
        """
        size = self.shape if size is None else size
        return position[0] * size[0] + position[1]

    def get_local_position(
            self,
            position: int,
            size: Tuple[int, int] = None
    ) -> Tuple[int, int]:
        """Get local position from a global position.

        Args:
            position (int): Global position.
            size (Tuple[int, int], optional): Size of the maze. Defaults to None.

        Returns:
            Tuple[int, int]: (x, y) coordinates
        """
        shape = self.shape[1] if size is None else size[1]
        column = position // shape
        row = position - (column * shape)
        return column, row

    def set_occlusion_on(self) -> None:
        """Set occlusion mask on."""
        self.occlusion = True

    def set_occlusion_off(self) -> None:
        """Set occlusion mask off."""
        self.occlusion = False

    def define_pathways(self, pathways: List[int]) -> List[int]:
        """Define pathways for the maze.

        Args:
            pathways (List[int]): List of visited nodes.

        Returns:
            List[int]: List of pathways.
        """
        _pathways = defaultdict(list)
        for start, end in pathways:
            _pathways[start].append(end)
        _pathways[self.get_global_position(self.end)] = []

        self.undirected_pathways = deepcopy(_pathways)
        pathways_dict = deepcopy(_pathways)
        for key, values in _pathways.items():
            for value in values:
                pathways_dict[value].append(key)
        return pathways_dict

    # TODO change to pygame dependency
    # TODO change key and door to have a transition function
    def render(self, mode: str = "human") -> list[float] | None:
        """Render the environment current state.

        Args:
            mode (str, optional): Mode to render the environment. Defaults to "human".
            Modes:
                Name        description
                human       render a view (image)
                rgb_array   render current state as a numpy array

        Raises:
            ResetException: If the environment is not reseted.

        Returns:
            Union[Any, bool, None]: Render view or numpy array.
        """
        if not self.reseted:
            raise ResetException("Please reset the environment first.")

        if self.render_utils is None:
            viewer = rendering.Viewer(self.screen_width, self.screen_height)
            self.render_utils = RenderUtils(self.shape, viewer) \
                .draw_start(self.start) \
                .draw_end(self.end) \
                .draw_agent(self.agent) \
                .draw_walls(self.maze)

        if self.occlusion:
            if self.render_utils is None:
                raise SettingsException("Viewer is not set")

            self.render_utils.draw_mask(self.create_mask())

        if self.key_and_door:
            if self.key is None or self.door is None:
                raise SettingsException("Door or key not set.")

            if self.render_utils is None:
                raise SettingsException("Viewer is not set")

            self.render_utils \
                .draw_key(self.key) \
                .draw_door(self.door)

        if self.icy_floor:
            if self.ice_floors is None:
                raise SettingsException("Ice floors not set.")

            if self.render_utils is None:
                raise SettingsException("Viewer not set.")

            ice_floors = [self.get_local_position(position) for position in self.ice_floors]
            self.render_utils \
                .draw_ice_floors(ice_floors)

        tile_h = self.render_utils.tile_h
        tile_w = self.render_utils.tile_w
        new_x = self.agent[1] * tile_w - self.start[1] * tile_w
        new_y = self.agent[0] * tile_h - self.start[0] * tile_h
        self.render_utils.agent_transition.set_translation(new_x, new_y)

        return self.render_utils.viewer.render(return_rgb_array=mode == "rgb_array")

    def translate_position(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert current position from original maze size to mask size.

        Args:
            position: Tuple[int, int]: (y, x) coordinates

        Returns:
            (y, x) coordinates
        """
        yoriginal, xoriginal = self.shape
        ymaze, xmaze = self.maze.shape
        width = int(position[1] / (xoriginal / xmaze))
        height = int(position[0] / (yoriginal / ymaze))
        return (height, width)

    # FIXME it has to change for ice floors
    def get_state(self) -> List[int] | ndarray:
        """
        Get the current state as a vector.

        Returns:
            state: List[int] with the current positions:
                0: agent global position
                1: start global position
                2: goal global position
                3+ maze structure in a vector
        """
        maze = self.maze if not self.occlusion else self.create_mask()
        maze = maze[1:-1, 1:-1]

        agent = self.get_global_position(
            self.translate_position(
                self.agent), maze.shape)
        start = self.get_global_position(
            self.translate_position(
                self.start), maze.shape)
        goal = self.get_global_position(
            self.translate_position(
                self.end), maze.shape)

        if self.key_and_door:
            key = self.get_global_position(
                self.translate_position(
                    self.key), maze.shape) if self.key else -1
            door = self.get_global_position(
                self.translate_position(
                    self.door), maze.shape) if self.door else -1

        if self.occlusion:
            tiles = [x for x in range(goal + 1) if x % 2 != 0]
            for tile in tiles:
                _neighbors = get_neighbors(tile, shape=maze.shape)
                _neighbors = [
                    self.get_local_position(neighbor, maze.shape) for _, neighbor in _neighbors
                ]
                values = np.array([maze[y, x] for y, x in _neighbors])
                if (values == 1).all():
                    height, width = self.get_local_position(tile, maze.shape)
                    maze[height, width] = 1

        maze = maze.flatten()
        maze[start] = 0
        maze[goal] = 0

        state = np.array([agent, start, goal])
        if self.key_and_door:
            state = np.hstack((state, [key, door]))

        state = np.hstack((state, maze))
        return state

    # TODO adapt reward function to be 1 - (-.1 / (self.shape[0] *
    # self.shape[1]) * len(shortest_path) )
    # FIXME if the agent is on the ice floor it should die
    def step(
        self,
        action: int
    ) -> Tuple[List[List[int]], float, bool, dict[str, List[int]]]:
        """Take a step in the environment.

        Args:
            action (int): Action to take.
            Actions:
                Name    Description
                0       UP
                1       RIGHT
                2       DOWN
                3       LEFT

        Raises:
            ActionException: If the action is not in the action space.

        Returns:
            Tuple[List[List[int]], float, bool, dict[str, List[int]]]: Gym step return.
        """
        if action not in self.actions:
            raise ActionException(
                f"Action should be in {self.actions.keys()} it was {action}")

        destiny = np.array(self.agent) + self.actions[action]
        agent_global_position = self.get_global_position(self.agent)
        destiny_global_position = self.get_global_position(destiny)
        if destiny_global_position in self.pathways[agent_global_position]:
            if self.key_and_door and (
                    np.array(tuple(destiny)) == self.door).all():
                if self.key is not None:
                    destiny = self.agent
                else:
                    self.door = None

            if self.key_and_door and (
                    np.array(tuple(destiny)) == self.key).all():
                self.key = None

            self.agent = tuple(destiny)

        self.step_count += 1
        done = (np.array(self.agent) == self.end).all(
        ) or self.step_count >= self.max_episode_steps
        reward = -.1 / (self.shape[0] * self.shape[1]
                        ) if not (np.array(self.agent) == self.end).all() else 1

        return self.get_state(), reward, done, {}

    def reset(self, agent: bool = True, render: bool = False) -> Union[List[int], ndarray]:
        """Reset the environment.

        Args:
            agent (bool, optional): If agent is True, reset the agent position and keep the maze.
                If agent is False, reset the maze and agent. Defaults to True.
            render (bool, optional): If should return a rendered view of the maze.
                Defaults to False.

        Returns:
            Union[List[int], ndarray]: State of the environment.
        """
        self.reseted = True
        self.step_count = 0
        self.render_utils = None

        if not agent or self.maze is None:
            if not self.icy_floor:
                self.maze, self._pathways = self._generate()
            else:
                self.maze, self._pathways = self._generate(min_paths=2)

            self.pathways = self.define_pathways(self._pathways)
            self.agent = self.start

        if self.key_and_door and self.door is None and self.key is None:
            self.door, self.key = self.set_key_and_door()

        if self.icy_floor and self.ice_floors is None:
            self.ice_floors = self.set_ice_floors()

        return self.get_state() if not render else self.render("rgb_array")

    def generate(self, path: str, amount: int = 1) -> None:
        """Generate a maze and save it to a file.

        Args:
            path (str): Path to save the maze (int, optional): Amount of mazes to generate. Defaults to 1.
        """
        for _ in range(amount):
            self.reset(agent=False)
            hash_idx = hash(self)
            if not os.path.exists(path):
                os.makedirs(path)

            file_path = f'{path}/{hash_idx}.txt'
            self.save(file_path)

    def close(self) -> None:
        """Close the environment. Note: This does not reset the environment."""
        if self.render_utils is not None:
            self.render_utils.viewer.close()
            self.render_utils = None

    # FIXME it has to save the ice floors
    def save(self, path: str) -> None:
        """Save the current maze separated by ';'.

        Args:
            path (str): Path to save the current maze

        File:
            Position    Description
            0           Maze paths
            1           Start position
            2           Goal position
        """
        file = path.split('/')[-1]
        path = '/'.join(path.split('/')[:-1])
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f'{path}/{file}', 'w', encoding="utf-8") as _file:
            if self.key_and_door:
                _file.write(
                    f'{self._pathways};{self.start};{self.end};{self.key};{self.door}')
            else:
                _file.write(f'{self._pathways};{self.start};{self.end}')

    # FIXME it has to load the ice floors
    def load(self, path: str) -> None:
        """
        Load the maze from a file.

        Args:
            path (str): Path to save the file
        """
        with open(path, 'r', encoding="utf-8") as _file:
            for line in _file:
                info = line

        try:
            visited, start, end, key, door = info.split(';')
            self.key = ast.literal_eval(key)
            self.door = ast.literal_eval(door)
        except ValueError:
            visited, start, end = info.split(';')

        pathways = ast.literal_eval(visited)
        self.start = ast.literal_eval(start)
        self.end = ast.literal_eval(end)

        self.maze, self._pathways = self._generate(visited=pathways)
        self.pathways = self.define_pathways(self._pathways)
        self.agent = self.start

        if self.key_and_door and self.key is None and self.door is None:
            self.door, self.key = self.set_key_and_door()

        return self.reset(agent=True, render=False)

    def create_mask(self) -> List[List[int]]:
        """Create mask for occlusion based on the agent current position and maze structure.

        Returns:
            List[List[int]]: mask of the given maze for occlusion
        """
        tiles = []
        for height in range(self.shape[0]):
            for width in range(self.shape[1]):
                tiles.append(np.array((height, width)))

        maze = self.maze
        mask = deepcopy(maze)

        for tile in tiles:
            if (tile == self.agent).all():
                continue
            if (tile == self.agent).any():
                if tile[1] == self.agent[1]:  # Vertical mask
                    agent_row = self.agent[0] * 2 + 1
                    target_row = tile[0] * 2 + 1
                    column = tile[1] * 2 + 1
                    lower_bound = agent_row if agent_row < target_row else target_row
                    upper_bound = agent_row if agent_row > target_row else target_row
                    if (maze[lower_bound:upper_bound + 1, column] == 1).any():
                        mask[target_row, column] = 1
                else:  # Horizontal mask
                    agent_column = self.agent[1] * 2 + 1
                    target_column = tile[1] * 2 + 1
                    row = tile[0] * 2 + 1
                    lower_bound = agent_column if agent_column < target_column else target_column
                    upper_bound = agent_column if agent_column > target_column else target_column
                    if (maze[row, lower_bound:upper_bound + 1] == 1).any():
                        mask[row, target_column] = 1
            else:  # Diagonal mask
                target_row, target_column = tile * 2 + 1
                agent_row, agent_column = np.array(self.agent) * 2 + 1

                column_lower_bound = agent_column
                column_upper_bound = target_column
                column = False
                if not agent_column < target_column:
                    column_lower_bound = target_column
                    column_upper_bound = agent_column
                    column = True

                row_lower_bound = agent_row
                row_upper_bound = target_row
                row = False
                if not agent_row < target_row:
                    row_lower_bound = target_row
                    row_upper_bound = agent_row
                    row = True

                matrix = maze[
                    row_lower_bound:row_upper_bound + 1,
                    column_lower_bound:column_upper_bound + 1
                ]

                if matrix.shape[0] == matrix.shape[1]:
                    identity = []
                    if row and column:
                        x_range = range(matrix.shape[0] - 1)
                        identity = [[x, x + 1] for x in x_range]
                    if not (row and column):
                        identity = [[x + 1, x]
                                    for x in range(matrix.shape[0])][:-1]
                    if not column and row:
                        x_range = range(matrix.shape[0] - 1, 0, -1)
                        y_range = range(1, matrix.shape[0])
                        identity = [[x, y] for x, y in zip(x_range, y_range)]
                    if column and not row:
                        x_range = range(matrix.shape[0] - 1)
                        y_range = range(matrix.shape[0] - 2, -1, -1)
                        identity = [[x, y] for x, y in zip(x_range, y_range)]

                    for idx in identity:
                        if matrix[idx[1], idx[0]] == 1:
                            mask[target_row, target_column] = 1
                            continue
                else:
                    mask[target_row, target_column] = 1

        return mask

    def solve(self, mode: str = 'shortest') -> List[Tuple[int, int]]:
        """Solve the current maze. For key and door the graph has to be directed, because
        the agent has to come back, while the others it doesn't.

        Args:
            mode (str, optional): Mode to solve. Defaults to 'shortest'.
            Name        Description
            shortest    returns the shortest path
            all         returns all paths

        Raises:
            ValueError: If mode is not 'shortest' or 'all'.
            ResetException: If environment was not reseted.

        Returns:
            List[List[Tuple[int, int]]]: List of paths
        """
        if not self.reseted:
            raise ResetException("Please reset the environment first.")

        if mode not in ['shortest', 'all']:
            raise ValueError("mode should be 'shortest' or 'all'")

        with RecursionLimit(100000):
            if self.key_and_door and self.key is not None and self.door is not None:
                self.dfs.set_key_and_door(
                    self.get_global_position(self.key),
                    self.get_global_position(self.door)
                )
            if self.key is not None:
                mode = "shortest"

            graph = self.pathways if self.key is not None else self.undirected_pathways
            paths = self.dfs.find_paths(graph, mode == "shortest")

        if mode == "shortest":
            return [[node.identifier for node in min(paths)]]
        else:
            numbered_paths = []
            for path in paths:
                numbered_paths.append([node.identifier for node in path])
            return numbered_paths

    def change_start_and_goal(
        self, min_distance: int = None
    ) -> Tuple[Tuple[int], Tuple[int]]:
        """Changes the start and goal of the maze to not be always at the bottom left and
        upper right corners.

        Args:
            min_distance (int, optional): how far the start and goal should be.
            If nothing is passed it uses (width + height) // 2. Defaults to None.

        Returns:
            start (Tuple[int, int]): (y, x) coordinates
            end (Tuple[int, int]): (y, x) coordinates
        """
        if min_distance is None:
            width, height = self.shape
            min_distance = (width + height) // 2

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

        end_idx = np.random.choice(list(range(len(possible_goals))), 1)[0]
        end = possible_goals[end_idx]
        self.start = tuple(start)
        self.end = tuple(end)
        self.agent = self.start

        self.dfs.start = self.get_global_position(self.start)
        self.dfs.end = self.get_global_position(self.end)
        return start, end

    def agent_random_position(self) -> None:
        """
        Put the agent in a random position of the maze. This is mostly used if you want
        to create a dataset with diverse positions for your agent.
        """
        self.reset()
        self.agent = (
            random.randint(0, self.shape[0] - 1),
            random.randint(0, self.shape[1] - 1)
        )

    def set_key_and_door(
            self,
            min_distance: int = None,
            count: int = 0
    ) -> Tuple[List[int], List[int]]:
        """Set the key and door in the maze. Not all mazes have the right structure to have
        key and door in the setting we want (key outside the path to the door), so sometimes
        we restart the maze to find a new structure that might handle this setting. This is a
        iffy solution at best, we should look into something that does not require a maze restart.

        Args:
            min_distance: int = min distance from start to door. If none is informed the 
            environment uses (Height + Width) // 2.

        Returns:
            door (Tuple[int, int]): (y, x) coordinates for the door.
            key (Tuple[Tuple[int, int]): (x, y) coordinates for the key.
        """
        avoid = [
            self.get_global_position(self.start),
            self.get_global_position(self.end)
        ]
        if min_distance is None:
            min_distance = (self.shape[0] + self.shape[1]) // 2

        paths = self.solve(mode='all')
        if len(paths) > 1:
            intersection = list(
                set(paths[0]).intersection(*map(set, paths[1:])))
        else:
            intersection = paths[0]

        door, distance = 0, 0
        while distance < min_distance:
            door = np.random.choice(intersection, 1)[0]
            distance = np.abs(np.array([0, 0]) -
                              self.get_local_position(door)).sum()
            distance = 0 if door in avoid else distance

        def find_all_childs(possible_tiles, node_list):
            initial_len = len(possible_tiles)
            for node in node_list:
                edges = [
                    edge for edge in self.dfs.nodes[node].edges if edge not in node_list]
                if len(edges) > 0:
                    for edge in edges:
                        if edge < door:
                            possible_tiles.append(edge)

            if initial_len < len(possible_tiles):
                return find_all_childs(possible_tiles, possible_tiles)
            return possible_tiles

        try:
            possible_positions = find_all_childs(
                [],
                paths[0][:paths[0].index(door)]
            )
            key = np.random.choice(possible_positions, 1)[0]
        except ValueError:
            if count > 100:
                self.maze = None
                self.reset()
            return self.set_key_and_door(min_distance, count + 1)
        return self.get_local_position(door), self.get_local_position(key)

    def __hash__(self) -> int:
        """
        Create a hash of the edges of the maze.

        Returns:
            hash (int): in order to have a consistent hash, it sorts the inner tuples (the edges), 
            and the tuples (the list of edges), and remove duplicates before hashing the maze.
        """
        pathways = sorted(map(sorted, self._pathways))
        pathways = tuple(set(map(tuple, pathways)))
        return hash(pathways)

    # TODO I think we should only use those that have difference between both paths
    # it lets we test when changing from training to validation.
    def set_ice_floors(self) -> List[int]:
        """_summary_

        Raises:
            NotImplementedError: _description_

        Returns:
            List[int]: _description_
        """
        paths = self.dfs.graph[self.dfs.end].d
        ice_floors_option1 = list(set(paths[0]).difference(set(paths[1])))
        ice_floors_option1 = [node.identifier for node in ice_floors_option1]
        ice_floors_option2 = list(set(paths[1]).difference(set(paths[0])))
        ice_floors_option2 = [node.identifier for node in ice_floors_option2]
        return ice_floors_option1 if len(ice_floors_option1) != 0 else ice_floors_option2
