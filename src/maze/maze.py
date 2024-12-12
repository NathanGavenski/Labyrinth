"""Maze environment."""
import ast
from collections import defaultdict
from copy import deepcopy
import os
import random
from typing import Any, List, Tuple, Optional, Union, Dict

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.utils import seeding
from gymnasium.error import DependencyNotInstalled
import numpy as np
from numpy import ndarray

from .utils import create_mask
from .utils import transform_edges_into_walls
from .utils import get_neighbors, DFS, RecursionLimit
from .utils import SettingsException, ResetException, ActionException
from .utils.render import RenderUtils


# TODO add gym logger
class Maze(gym.Env[np.ndarray, Union[int, np.ndarray]]):
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
        Type: Discrete(4)
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
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    actions = {
        0: (1, 0),
        1: (0, 1),
        2: (-1, 0),
        3: (0, -1)
    }

    # env
    done = False
    state = None
    reseted = False

    # maze
    dfs = None
    maze = None
    undirected_pathways = None
    agent_transition = None
    pathways = None
    _pathways = None
    step_count = 0

    ## extra modes
    key = None
    door = None
    ice_floors = None

    # pygame
    clock = None
    screen = None
    render_utils = None

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
            visual: bool = False,
            render_mode: Optional[str] = None
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
        self.visual = visual
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_episode_steps = max_episode_steps
        self.start = start
        self.end = (self.shape[0] - 1, self.shape[1] - 1) if end is None else end
        self.agent = self.start
        self.render_mode = render_mode

        self.seed()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 255, (screen_width, screen_height, 3), np.uint8)

        self.occlusion = occlusion
        self.key_and_door = key_and_door
        self.icy_floor = icy_floor

        if sum([self.occlusion, self.key_and_door, self.icy_floor]) > 1:
            raise SettingsException("Different modes cannot be active at the same time.")

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
        max_paths: int = None,
        random_amount: int = 0,
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

            self.dfs = DFS(
                edges,
                maze.shape,
                start=start,
                end=end,
                random_amount=random_amount
            )
            with RecursionLimit(100000):
                graph = self.dfs.generate_path(
                    min_paths=min_paths,
                    max_paths=max_paths,
                )

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

        for key, values in pathways_dict.items():
            pathways_dict[key] = list(set(values))
        return pathways_dict

    def render(self) -> Union[List[float], None]:
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

        if self.render_mode is None:
            assert self.spec is not None
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled("pygame is not installed, run `pip install pygame`") from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )

                if self.clock is None:
                    self.clock = pygame.time.Clock()

            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.render_utils is None:
            self.render_utils = RenderUtils(self.shape, self.screen)

            if self.icy_floor:
                if self.ice_floors is None:
                    raise SettingsException("Ice floors not set.")

                if self.render_utils is None:
                    raise SettingsException("Viewer not set.")

        self.render_utils \
            .redraw() \
            .draw_end(self.end) \
            .draw_start(self.start) \
            .draw_agent(self.agent)

        if self.key_and_door:
            self.render_utils.draw_key(self.key).draw_door(self.door)

        if self.ice_floors:
            self.render_utils.draw_ice_floors(self.ice_floors)

        self.render_utils.draw_walls(self.maze)

        if self.occlusion:
            mask = create_mask(self.shape, self.maze, self.agent)
            self.render_utils.draw_mask(mask)

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

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

    def get_state(self) -> Union[List[int], ndarray]:
        """Get the current state as a vector.

        Returns:
            state: List[int] with the current positions:
                0: agent global position
                1: start global position
                2: goal global position
                3+ maze structure in a vector
        """
        if self.render_mode == "rgb_array":
            import pyglet
            try:
                pyglet.options["headless"] = True
                return self.render().copy()
            finally:
                pyglet.options["headless"] = False

        maze = self.maze
        if self.occlusion:
            maze = create_mask(self.shape, self.maze, self.agent)
        maze = maze[1:-1, 1:-1]

        agent = self.get_global_position(self.translate_position(self.agent), maze.shape)
        start = self.get_global_position(self.translate_position(self.start), maze.shape)
        goal = self.get_global_position(self.translate_position(self.end), maze.shape)

        if self.key_and_door:
            key = self.get_global_position(
                self.translate_position(self.key),
                maze.shape
            ) if self.key else -1
            door = self.get_global_position(
                self.translate_position(self.door),
                maze.shape
            ) if self.door else -1

        if self.icy_floor:
            ice_floors = [self.translate_position(ice) for ice in self.ice_floors]
            ice_floors = [self.get_global_position(ice, maze.shape) for ice in ice_floors]

        if self.occlusion:
            goal = self.get_global_position(self.translate_position(self.end), maze.shape)
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
        if self.icy_floor:
            state = np.hstack((state, ice_floors))

        state = np.hstack((state, maze))
        return state

    # TODO adapt reward function to be 1 - (-.1 / (self.shape[0] *
    # self.shape[1]) * len(shortest_path) )
    def step(
        self,
        action: int
    ) -> Tuple[List[List[int]], float, bool, Dict[str, List[int]]]:
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
        if isinstance(action, np.ndarray) and len(action.shape) == 0:
            action = action.item()

        if action not in self.actions:
            raise ActionException(f"Action should be in {self.actions.keys()} it was {action}")

        if self.done:
            logger.warn(
                "You are calling 'step()' even though this environment has already finished. "
                "You should always call 'reset()' after finishing the environment."
            )
            raise ActionException("Episode is already finished.")

        destiny = np.array(self.agent) + self.actions[action]
        agent_global_position = self.get_global_position(self.agent)
        destiny_global_position = self.get_global_position(destiny)
        if destiny_global_position in self.pathways[agent_global_position]:
            if self.key_and_door and (np.array(tuple(destiny)) == self.door).all():
                if self.key is not None:
                    destiny = self.agent
                else:
                    self.door = None

            if self.key_and_door and (
                    np.array(tuple(destiny)) == self.key).all():
                self.key = None

            self.agent = tuple(destiny)
        self.step_count += 1
        done = (np.array(self.agent) == self.end).all()
        terminated = self.step_count >= self.max_episode_steps
        if self.icy_floor and tuple(destiny) in self.ice_floors:
            reward = -100
            done = True
        else:
            if not (np.array(self.agent) == self.end).all():
                reward = -.1 / (self.shape[0] * self.shape[1])
            else:
                reward = 1

        self.done = done or terminated
        return self.get_state(), reward, done, terminated, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Union[List[int], ndarray]:
        """Reset the environment.

        Args:
            agent (bool, optional): If agent is True, reset the agent position and keep the maze.
                If agent is False, reset the maze and agent. Defaults to True.
            render (bool, optional): If should return a rendered view of the maze.
                Defaults to False.

        Returns:
            Union[List[int], ndarray]: State of the environment.
        """
        agent = True if options is None or "agent" not in options.keys() else options["agent"]
        self.seed(seed)

        if self.render_utils is not None:
            import pygame

            self.screen = None
            self.render_utils = None
            del self.screen
            del self.render_utils
            pygame.display.quit()
            pygame.quit()

        self.reseted = True
        self.step_count = 0
        self.render_utils = None
        self.done = False

        if not agent or self.maze is None:
            if self.icy_floor:
                self.maze, self._pathways = self._generate(min_paths=2, random_amount=25)
            elif self.key_and_door:
                self.maze, self._pathways = self._generate(max_paths=1)
            else:
                self.maze, self._pathways = self._generate(random_amount=25)

            self.pathways = self.define_pathways(self._pathways)
        self.agent = self.start

        if self.key_and_door and self.door is None and self.key is None:
            try:
                self.door, self.key = self.set_key_and_door()
            except SettingsException:
                self.maze = None
                return self.reset()

        if self.icy_floor and self.ice_floors is None:
            self.ice_floors = self.set_ice_floors()

        return self.render() if self.render_mode == "rgb_array" else self.get_state(), {}

    def generate(self, path: str, amount: int = 1) -> None:
        """Generate a maze and save it to a file.

        Args:
            path (str): Path to save the maze (int, optional): Amount of mazes to generate.
            Defaults to 1.
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
            import pygame

            self.screen = None
            self.render_utils = None
            del self.screen
            del self.render_utils
            pygame.display.quit()
            pygame.quit()

    def save(self, path: str) -> None:
        """Save the current maze separated by ';'.

        Args:
            path (str): Path to save the current maze

        File:
            Position    Description
            0           Maze paths
            1           Start position
            2           Goal position
            3           Key | Ice floors position
            4           Door position
        """
        if "/" in path:
            file = path.split('/')[-1]
            path = '/'.join(path.split('/')[:-1])
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            file = path
            path = "."

        pathways = self._pathways
        if isinstance(pathways, dict):
            pathways = []
            for tile, edges in self._pathways.items():
                for edge in edges:
                    pathways.append((tile, edge))

        with open(f'{path}/{file}', 'w', encoding="utf-8") as _file:
            save_string = f"{pathways};{self.start};{self.end}"
            if self.key_and_door:
                save_string += f";{self.key};{self.door}"
            if self.icy_floor:
                save_string += f";{self.ice_floors}"
            _file.write(save_string)

    def load(self, path: str) -> Union[List[int], ndarray]:
        """Load the maze from a file.

        Args:
            path (str): Path to save the file
        """
        with open(path, 'r', encoding="utf-8") as _file:
            for line in _file:
                info = line

        visited, start, end, *misc = info.split(";")
        if self.key_and_door:
            key, door = misc[0], misc[1]
            self.key = ast.literal_eval(key)
            self.door = ast.literal_eval(door)
        if self.icy_floor:
            ice_floors = misc[0]
            self.ice_floors = ast.literal_eval(ice_floors)

        pathways = ast.literal_eval(visited)
        self.start = ast.literal_eval(start)
        self.end = ast.literal_eval(end)

        self.maze, self._pathways = self._generate(visited=pathways)
        self.pathways = self.define_pathways(self._pathways)
        self.agent = self.start

        if self.key_and_door and self.key is None and self.door is None:
            self.door, self.key = self.set_key_and_door()

        return self.reset(options={"agent": True})

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
            if self.icy_floor and self.ice_floors is not None:
                mode = "all"

            paths = self.dfs.find_paths(self.pathways, mode == "shortest")

        if self.icy_floor and self.ice_floors is not None:
            ice_floors = set([self.get_global_position(floor) for floor in self.ice_floors])
            _paths = []
            for path in paths:
                if len(set(path).intersection(ice_floors)) == 0:
                    _paths.append(path)
            paths = _paths

        if mode == "shortest":
            return [[node.identifier for node in min(paths)]]

        numbered_paths = []
        for path in paths:
            numbered_paths.append([node.identifier for node in path])
        return numbered_paths

    def change_start_and_goal(self, min_distance: int = None) -> Tuple[Tuple[int], Tuple[int]]:
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
        """Put the agent in a random position of the maze. This is mostly used if you want
        to create a dataset with diverse positions for your agent.
        """
        self.reset()
        self.agent = (
            random.randint(0, self.shape[0] - 1),
            random.randint(0, self.shape[1] - 1)
        )

    def set_key_and_door(self) -> Tuple[List[int], List[int]]:
        """Set the key and door in the maze. Not all mazes have the right structure to have
        key and door in the setting we want (key outside the path to the door), so sometimes
        we restart the maze to find a new structure that might handle this setting. This is a
        iffy solution at best, we should look into something that does not require a maze restart.

        Returns:
            door (Tuple[int, int]): (y, x) coordinates for the door.
            key (Tuple[Tuple[int, int]): (x, y) coordinates for the key.

        Raises:
            SettingsException: if there are no possible candidates for key and door, it raises
            and exception to reset the maze to look for another possible structure that suffices
            key and door setting.
        """
        paths = self.solve(mode='all')
        if len(paths) > 1:
            intersection = list(set(paths[0]).intersection(*map(set, paths[1:])))
        else:
            intersection = paths[0]

        key_tiles = []
        door_tiles = []
        for node in self.dfs.graph.values():
            if node not in intersection:
                key_tiles.append(node)
            if node in intersection and len(node.edges) > 1:
                for edge in node.edges:
                    if edge not in [self.dfs.start, self.dfs.end]:
                        if edge in intersection and len(node.visited_edges) == 1:
                            door_tiles.append(edge)
        if len(door_tiles) == 0:
            raise SettingsException("No possible candidate for door or key")

        door = door_tiles[-1]
        key_tiles = [node for node in key_tiles if door not in node.d[0]]
        if len(key_tiles) == 0:
            raise SettingsException("No possible candidate for door or key")
        key = random.choice(key_tiles)

        return self.get_local_position(door.identifier), self.get_local_position(key.identifier)

    def __hash__(self) -> int:
        """Create a hash of the edges of the maze.

        Returns:
            hash (int): in order to have a consistent hash, it sorts the inner tuples (the edges), 
            and the tuples (the list of edges), and remove duplicates before hashing the maze.
        """
        pathways = sorted(map(sorted, self._pathways))
        pathways = tuple(set(map(tuple, pathways)))
        return hash(pathways)

    def set_ice_floors(self) -> List[int]:
        """Set ice floors in the maze. Ice floors are tiles that the agent can slide through

        Returns:
            List[int]: List of ice floors.
        """
        paths = self.solve(mode="all")
        paths.sort(key=len)
        for longer_path in paths[::-1]:
            for smaller_path in paths:
                ice_floors_option = list(set(smaller_path).difference(set(longer_path)))
                if len(ice_floors_option) != 0:
                    ice_floors = ice_floors_option
                    break
            else:
                continue
            break
        return [self.get_local_position(position) for position in ice_floors]
