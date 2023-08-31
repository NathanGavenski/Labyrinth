"""Unit tests for the Maze environment."""
import ast
import os
from typing import Tuple
import unittest
import shutil

import gym
import numpy as np
from PIL import Image
import pytest

# pylint: disable=[W0611]
from src import maze
from src.create_expert import state_to_action
from src.maze.utils.utils import SettingsException


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
    x_position = int(position[1] / (x_original/x_maze))
    y_position = int(position[0] / (y_original/y_maze))
    return (y_position, x_position)


class TestCases(unittest.TestCase):
    """Test cases for the Maze environment."""

    test_files_path = "./src/tests/assets/"

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the test environment."""
        cls.env = None
        if not os.path.exists("./tests/"):
            os.makedirs("./tests/")

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down the test environment."""
        shutil.rmtree("./tests/")
        cls.env.close()

    def tearDown(self) -> None:
        """Tear down after every test function."""
        TestCases.env.close()

    def test_init(self):
        """Test the initialization of the environment."""
        TestCases.env = env = gym.make("Maze-v0", shape=(3, 3), occlusion=False)
        state = env.reset()

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

    # pylint: disable=[W0212, protected-access]
    def test_save(self):
        """Test the save function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.reset()
        env.save("./tests/test.txt")

        assert os.path.exists("./tests/test.txt")

        with open("./tests/test.txt", encoding="utf-8") as _file:
            for line in _file:
                info = line

        visited, start, end = info.split(";")
        assert env._pathways == ast.literal_eval(visited)
        assert env.start == ast.literal_eval(start)
        assert env.end == ast.literal_eval(end)

    def test_load(self):
        """Test the load function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.load("./src/tests/assets/structure_test.txt")

        with open("./src/tests/assets/structure_test.txt", encoding="utf-8") as _file:
            for line in _file:
                info = line

        visited, start, end = info.split(";")
        assert env._pathways == ast.literal_eval(visited)
        assert env.start == ast.literal_eval(start)
        assert env.end == ast.literal_eval(end)

        img = np.array(Image.open("./src/tests/assets/render_test.png"))
        assert (env.render("rgb_array") == img).all()

    def test_step(self):
        """Test the step function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        state = env.load("./src/tests/assets/structure_test.txt")
        assert state[0] == 0

        state, _, _, _ = env.step(0)
        assert state[0] == 38
        assert env.agent == (1, 0)

    def test_step_wall(self):
        """Test the step function when the agent tries to move into a wall."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        state = env.load("./src/tests/assets/structure_test.txt")
        assert state[0] == 0

        state, _, _, _ = env.step(1)
        assert state[0] == 0

    def test_reset(self):
        """Test the reset function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        state = env.load("./src/tests/assets/structure_test.txt")
        assert state[0] == 0

        state, _, _, _ = env.step(0)
        assert state[0] == 38

        state = env.reset()
        assert state[0] == 0

    def test_global_position(self):
        """Test the get_global_position function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        assert env.get_global_position((9, 9), (10, 10)) == get_global_position((9, 9), (10, 10))

    def test_local_position(self):
        """Test the get_local_position function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.reset()
        end = env.get_global_position((9, 9), (10, 10))
        assert (9, 9) == env.get_local_position(end)

    def test_change_start_and_goal(self):
        """Test the change_start_and_goal function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.load("./src/tests/assets/structure_test.txt")
        env.change_start_and_goal()
        assert env.start != (0, 0)

    def test_agent_random_position(self):
        """Test the agent_random_position function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.load("./src/tests/assets/structure_test.txt")
        agent_initial_position = env.agent
        env.agent_random_position()
        assert agent_initial_position != env.agent

    def test_size(self):
        """Test the size function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(5, 5), occlusion=False)
        state = env.reset()

        maze_size = (5, 5)
        maze_size = ((maze_size[0] * 2) - 1) ** 2
        assert maze_size + 3 == state.shape[0]

    def test_occlusion(self):
        """Test the occlusion function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=True)
        env.load("./src/tests/assets/structure_test.txt")
        state = env.render("rgb_array")
        test_state = np.array(Image.open("./src/tests/assets/occlusion_test.png"))
        assert (state == test_state).all()

        env.close()

        TestCases.env = env = gym.make("Maze-v0", shape=(3, 3), occlusion=True)
        state = env.load("./src/tests/assets/occlusion_vector_test.txt")
        with open("./src/tests/assets/vector_occlusion_test.txt", encoding="utf-8") as _file:
            for line in _file:
                test_state = line
        assert (state == ast.literal_eval(test_state)).all()

    def test_key_and_door(self):
        """Test the key_and_door function."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        state = env.load("./src/tests/assets/key_and_door_test.txt")

        with open("./src/tests/assets/key_and_door_test.txt", 'r', encoding="utf-8") as _file:
            for line in _file:
                info = line

        _, _, _, key, door = info.split(';')
        key = ast.literal_eval(key)
        door = ast.literal_eval(door)

        maze_size = env.shape
        maze_size = ((maze_size[0] * 2) - 1) ** 2

        test_render = np.array(Image.open(f"{self.test_files_path}render_key_and_door_test.png"))

        assert door == env.door
        assert key == env.key
        assert state[3] == 2
        assert state[4] == 316
        assert state.shape[0] == maze_size + 5
        assert (env.render("rgb_array") == test_render).all()

    def test_solve_shortest(self):
        """Test the solve function with the shortest option."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.load("./src/tests/assets/structure_test.txt")
        solve = env.solve()
        with open("./src/tests/assets/solve_test.txt", encoding="utf-8") as _file:
            for line in _file:
                test_solve = line

        assert solve == ast.literal_eval(test_solve)
        env.close()

        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        env.load("./src/tests/assets/key_and_door_test.txt")

        with open("./src/tests/assets/solve_key_and_door.txt", 'r', encoding="utf-8") as _file:
            for line in _file:
                test_solve = line
        assert (env.solve(mode="shortest") == ast.literal_eval(test_solve)).all()

    def test_solve_all(self):
        """Test the solve function with the all option."""
        TestCases.env = env = gym.make("Maze-v0", shape=(100, 100), key_and_door=True)
        env.load("./src/tests/assets/structure_solve_all_test.txt")
        first_solutions = env.solve("all")

        env.close()
        TestCases.env = env = gym.make("Maze-v0", shape=(100, 100), key_and_door=False)
        env.load("./src/tests/assets/structure_solve_all_test.txt")
        second_solutions = env.solve("all")

        for _x, _y in zip(first_solutions, second_solutions):
            assert len(_x) > len(_y)
            assert (set(_y) - set(_x)) == set()

    def test_door(self):
        """Test the key and door setting without a key (the agent can't open the door)."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        state = env.reset()
        door = env.get_global_position(env.door)

        env.key_and_door = False
        path = env.solve("shortest")
        env.key_and_door = True

        agent = None
        for idx, tile in enumerate(path):
            if idx < len(path) - 1:
                action = state_to_action(tile, path[idx + 1], shape=(10, 10))
                if path[idx + 1] == door:
                    agent = state[0]
                state, _, _, _ = env.step(action)

                if agent is not None:
                    assert agent == state[0]
                    break

        assert agent is not None

    def test_key(self):
        """Test the key and door setting with a key (the agent can open a door)."""
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), key_and_door=True)
        env.reset()
        path = env.solve("shortest")

        for idx, tile in enumerate(path):
            if idx < len(path) - 1:
                action = state_to_action(tile, path[idx + 1], shape=(10, 10))
                env.step(action)

        assert env.agent == env.end

    def test_key_and_door_occlusion(self):
        """Test if environment allows 'Key and Door' and 'Occlusion' at the same time."""
        with pytest.raises(SettingsException) as excinfo:
            gym.make("Maze-v0", shape=(10, 10), key_and_door=True, occlusion=True)

        assert "Both modes cannot be active at the same time." in str(excinfo.value)


    def test_icy_floor(self):
        """Test the icy floor setting."""
        # Create icy floor environment
        # Validate the first sate
        # Validate trajectory
        raise NotImplementedError()

    def test_break_floor(self):
        """Test if the icy floor breaks when the agent steps on it."""
        # Create icy floor environment
        # Validate the first state
        # Break the floor and fall
        raise NotImplementedError()
