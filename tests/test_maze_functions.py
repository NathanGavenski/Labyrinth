"""Unit tests for the Maze environment."""
import ast
import os
from typing import Tuple
import unittest
import shutil

import gymnasium as gym
import numpy as np
from PIL import Image
import pytest

from src import maze
from create_expert import state_to_action
from maze.utils.utils import SettingsException, ActionException
from src.maze.file_utils import create_file_from_environment, convert_from_file
from src.maze.interp import Interpreter

github = os.getenv("SERVER")
github = bool(int(github)) if github is not None else False
PATH = "/".join(__file__.split("/")[:-1])
TMP_PATH = f"{PATH}/tmp/"

def get_global_position(position: Tuple[int, int], size: Tuple[int] = (10, 10)) -> int:
    """Get the global position of a tile in the maze.

    Args:
        position (List[int]): Local position of the tile -- (x, y) coordinates.
        size (List[int], optional): Maze width and height. Defaults to [10, 10].

    Returns:
        int: Global position of the tile.
    """
    return position[0] * size[0] + position[1]


def translate_position(
    position: Tuple[int, int],
    shape: Tuple[int, int],
    maze_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """Translate the position of a tile in the maze.

    Args:
        position (Tuple[int, int]): Original position.
        shape (Tuple[int, int]): Original shape of the maze.
        maze_shape (Tuple[int, int]): New shape of the maze.

    Returns:
        position Tuple[int, int]: Translated position.
    """
    y_original, x_original = shape
    y_maze, x_maze = maze_shape
    x_position = int(position[1] / (x_original / x_maze))
    y_position = int(position[0] / (y_original / y_maze))
    return (y_position, x_position)


class TestCases(unittest.TestCase):
    """Test cases for the Maze environment."""

    test_files_path = "./assets/"

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment."""
        cls.env = None
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down the test environment."""
        shutil.rmtree(TMP_PATH)
        if cls.env is not None:
            cls.env.close()

    def tearDown(self) -> None:
        """Tear down after every test function."""
        if TestCases.env is not None:
            TestCases.env.close()

    def test_init(self):
        """Test the initialization of the environment."""
        TestCases.env = env = gym.make("Maze-v0", shape=(3, 3))
        state, info = env.reset()

        maze_size = env.shape
        maze_size = ((maze_size[0] * 2) - 1) ** 2
        assert maze_size + 3 == state.shape[0]

        maze_size = [((env.shape[0] * 2) - 1), ((env.shape[1] * 2) - 1)]
        agent = translate_position(env.agent, env.shape, maze_size)
        agent = get_global_position(agent, maze_size)
        assert state[0] == agent

        start = translate_position(env.start, env.shape, maze_size)
        start = get_global_position(start, maze_size)
        assert state[1] == start

        end = translate_position(env.end, env.shape, env.maze.shape)
        end = get_global_position(end, maze_size)
        assert state[2] == end

    def test_save(self):
        """Test the save function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        env.reset()
        create_file_from_environment(env, f"{TMP_PATH}test.maze")
        assert os.path.exists(f"{TMP_PATH}/test.txt")

        info, _ = convert_from_file(f"{TMP_PATH}test.maze")
        visited, start, end = info.split(";")
        visited = ast.literal_eval(visited)
        assert {frozenset(t) for t in env._pathways} == {frozenset(t) for t in visited}
        assert env.start == tuple(ast.literal_eval(start))
        assert env.end == tuple(ast.literal_eval(end))

    def test_save_key_and_door(self):
        """Test the save function with different variables."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        env.reset()
        create_file_from_environment(env, f"{TMP_PATH}key_and_door.maze")
        _, variables = convert_from_file(f"{TMP_PATH}key_and_door.maze")
        assert variables["key_and_lock"]

    def test_save_occlusion(self):
        """Test the save function with occlusion."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=True)
        env.reset()
        create_file_from_environment(env, f"{TMP_PATH}occlusion.maze")
        _, variables = convert_from_file(f"{TMP_PATH}occlusion.maze")
        assert variables["occlusion"]

    def test_save_icy_floor(self):
        """Test the save function with icy floor."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), icy_floor=True)
        env.reset()
        create_file_from_environment(env, f"{TMP_PATH}icy_floor.maze")
        _, variables = convert_from_file(f"{TMP_PATH}icy_floor.maze")
        assert variables["icy_floor"]

    def test_load(self):
        """Test the load function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        env.load(*convert_from_file(f"{PATH}/assets/structure_test.maze"))

        interpreter = Interpreter()
        with open(f"{PATH}/assets/structure_test.maze", encoding="utf-8") as _file:
            for line in _file:
                interpreter.eval(line)

        visited, start, end = convert_from_file(f"{PATH}/assets/structure_test.maze")[0].split(";")
        assert env._pathways == ast.literal_eval(visited)
        assert env.start == ast.literal_eval(start)
        assert env.end == ast.literal_eval(end)

    def test_step(self):
        """Test the step function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        state, _ = env.load(*convert_from_file(f"{PATH}/assets/structure_test.maze"))
        assert state[0] == 0

        state, *_ = env.step(0)
        assert state[0] == 38
        assert env.agent == (1, 0)

    def test_step_wall(self):
        """Test the step function when the agent tries to move into a wall."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        state, _ = env.load(*convert_from_file(f"{PATH}/assets/structure_test.maze"))
        assert state[0] == 0

        state, *_ = env.step(1)
        assert state[0] == 0

    def test_reset(self):
        """Test the reset function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        state, _ = env.load(*convert_from_file(f"{PATH}/assets/structure_test.maze"))
        assert state[0] == 0

        state, *_ = env.step(0)
        assert state[0] == 38.0

        state, _ = env.reset()
        assert state[0] == 0

    def test_global_position(self):
        """Test the get_global_position function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        assert env.get_global_position((9, 9), (10, 10)) == get_global_position((9, 9), (10, 10))

    def test_local_position(self):
        """Test the get_local_position function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        env.reset()
        end = env.get_global_position((9, 9), (10, 10))
        assert (9, 9) == env.get_local_position(end)

    def test_change_start_and_goal(self):
        """Test the change_start_and_goal function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        env.load(*convert_from_file(f"{PATH}/assets/structure_test.maze"))
        env.change_start_and_goal()
        assert env.start != (0, 0)

    def test_agent_random_position(self):
        """Test the agent_random_position function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        env.load(*convert_from_file(f"{PATH}/assets/structure_test.maze"))
        agent_initial_position = env.agent
        env.agent_random_position()
        assert agent_initial_position != env.agent

    def test_size(self):
        """Test the size function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(5, 5))
        state, _ = env.reset()

        maze_size = (5, 5)
        maze_size = ((maze_size[0] * 2) - 1) ** 2
        assert maze_size + 3 == state.shape[0]

    @pytest.mark.skipif(github, reason="render looks different on github")
    def test_occlusion_render(self):
        """Test the occlusion function."""
        TestCases.env = env = gym.make(
            "Maze-v0",
            shape=(10, 10),
            occlusion=True,
            render_mode="rgb_array"
        )
        env.load(*convert_from_file(f"{PATH}/assets/structure_test.maze"))
        env.occlusion = True
        state = env.render()
        test_state = np.array(Image.open(f"{PATH}/assets/occlusion_test.png"))
        assert (state == test_state).all()
        env.close()

    def test_occlusion_vector(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(3, 3), occlusion=True)
        state, _ = env.load(*convert_from_file(f"{PATH}/assets/occlusion_vector_test.maze"))
        with open(f"{PATH}/assets/vector_occlusion_test.txt", encoding="utf-8") as _file:
            for line in _file:
                test_state = line
        assert (state == ast.literal_eval(test_state)).all()

    def test_key_and_door(self):
        """Test the key_and_door function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        structure, variables = convert_from_file(f"{PATH}/assets/key_and_door_test.maze")
        state, info = env.load(structure, variables)

        _, _, _, key, door = structure.split(';')
        key = ast.literal_eval(key)
        door = ast.literal_eval(door)

        maze_size = env.shape
        maze_size = ((maze_size[0] * 2) - 1) ** 2

        assert door == env.door
        assert key == env.key
        assert state[3] == 2
        assert state[4] == 316
        assert state.shape[0] == maze_size + 5

    def test_solve_shortest(self):
        """Test the solve function with the shortest option."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10))
        env.load(*convert_from_file(f"{PATH}/assets/structure_test.maze"))
        solve = env.solve()
        with open(f"{PATH}/assets/solve_test.txt", encoding="utf-8") as _file:
            for line in _file:
                test_solve = line

        assert solve[0] == ast.literal_eval(test_solve)
        env.close()

        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        env.load(*convert_from_file(f"{PATH}/assets/key_and_door_test.maze"))

        with open(f"{PATH}/assets/solve_key_and_door.txt", 'r', encoding="utf-8") as _file:
            for line in _file:
                test_solve = line
        assert env.solve(mode="shortest")[0] == ast.literal_eval(test_solve)

    def test_solve_all(self):
        """Test the solve function with the all option."""
        TestCases.env = env = gym.make("Maze-v0", shape=(5, 5))
        env.reset()
        create_file_from_environment(env, f"{TMP_PATH}/tmp.maze")
        first_solutions = env.solve("all")
        first_solutions.sort(key=len)

        env.close()
        TestCases.env = env = gym.make("Maze-v0", shape=(5, 5))
        env.load(*convert_from_file(f"{TMP_PATH}/tmp.maze"))
        env.reset()
        second_solutions = env.solve("all")
        second_solutions.sort(key=len)

        first = sorted(sorted(sublist) for sublist in first_solutions)
        second = sorted(sorted(sublist) for sublist in second_solutions)

        for _x, _y in zip(first, second):
            assert len(_x) == len(_y)
            assert (set(_y) - set(_x)) == set()

    def test_door(self):
        """Test the key and door setting without a key (the agent can't open the door)."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        state = env.reset()
        door = env.get_global_position(env.door)

        env.key_and_door = False
        env.dfs.key_and_door = False
        path = env.solve("shortest")[0]
        env.key_and_door = True
        env.dfs.key_and_door = True

        agent = None
        for idx, tile in enumerate(path):
            if idx < len(path) - 1:
                action = state_to_action(tile, path[idx + 1], shape=(10, 10))
                if path[idx + 1] == door:
                    agent = state[0]
                state, *_ = env.step(action)

                if agent is not None:
                    assert agent == state[0]
                    break
        assert agent is not None

    def test_key(self):
        """Test the key and door setting with a key (the agent can open a door)."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        env.reset()
        path = env.solve("shortest")[0]

        for idx, tile in enumerate(path):
            if idx < len(path) - 1:
                action = state_to_action(tile, path[idx + 1], shape=(10, 10))
                env.step(action)

        assert env.agent == env.end

    def test_more_than_one_mode(self):
        """Test if environment allows 'Key and Door' and 'Occlusion' at the same time."""
        with pytest.raises(SettingsException) as excinfo:
            gym.make("Maze-v0", shape=(10, 10), key_and_door=True, occlusion=True)

        assert "Different modes cannot be active at the same time." in str(excinfo.value)

    def test_icy_floor(self):
        """Test the icy floor setting."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), icy_floor=True)
        state, info = env.reset()
        floors = []
        for floor in env.ice_floors:
            floor = env.translate_position(floor)
            floor = env.get_global_position(floor, env.maze[1:-1, 1:-1].shape)
            floors.append(floor)

        assert (state[3:len(floors) + 3] == floors).all()

        ice_floor = [env.get_global_position(floor) for floor in env.ice_floors]
        path = env.dfs.find_path(
            env.undirected_pathways,
            env.dfs.start,
            ice_floor[0],
            early_stop=True
        )
        path = [node.identifier for node in path[ice_floor[0]].d]
        intersec = list(set(path).intersection(set(ice_floor)))
        index = min([path.index(floor) for floor in intersec])
        path = path[:index + 1]

        for idx, tile in enumerate(path):
            if idx < len(path) - 1:
                action = state_to_action(tile, path[idx + 1], shape=(10, 10))
                state, _, done, _, _ = env.step(action)
        assert done

        with pytest.raises(ActionException):
            env.step(env.action_space.sample())
