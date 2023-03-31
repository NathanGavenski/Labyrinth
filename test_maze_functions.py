import ast
import os
import time
from typing import List
import unittest
import shutil 

import gym
import maze
import numpy as np
from PIL import Image

# occlusion option

def get_global_position(position: List[int], size: List[int] = [10, 10]) -> int:
    '''
    Get global position from a tile.
    '''
    return position[0] * size[0] + position[1]

def translate_position(position, shape, maze_shape):
    yoriginal, xoriginal = shape
    ymaze, xmaze = maze_shape
    x = int(position[1] / (xoriginal/xmaze))
    y = int(position[0] / (yoriginal/ymaze))
    return (y, x)

class TestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = None
        if not os.path.exists("./tests/"):
            os.makedirs("./tests/")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree("./tests/")
        cls.env.close()

    def test_init(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
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
        
    def test_save(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.reset()
        env.save("./tests/test.txt")

        assert os.path.exists("./tests/test.txt")

        with open("./tests/test.txt") as f:
            for line in f:
                info = line
        
        visited, start, end = info.split(";")
        assert env._pathways == ast.literal_eval(visited)
        assert env.start == ast.literal_eval(start)
        assert env.end == ast.literal_eval(end)

    def test_load(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.load("./maze/environment/utils/test/structure_test.txt")

        with open("./maze/environment/utils/test/structure_test.txt") as f:
            for line in f:
                info = line
            
        visited, start, end = info.split(";")
        assert env._pathways == ast.literal_eval(visited)
        assert env.start == ast.literal_eval(start)
        assert env.end == ast.literal_eval(end)

        img = np.array(Image.open("./maze/environment/utils/test/render_test.png"))
        assert (env.render("rgb_array") == img).all()

    def test_step(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        state = env.load("./maze/environment/utils/test/structure_test.txt")
        assert state[0] == 0

        state, _, _, _ = env.step(0)
        assert state[0] == 38
        assert env.agent == (1, 0)

    def test_step_wall(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        state = env.load("./maze/environment/utils/test/structure_test.txt")
        assert state[0] == 0

        state, _, _, _ = env.step(1)
        assert state[0] == 0

    def test_reset(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        state = env.load("./maze/environment/utils/test/structure_test.txt")
        assert state[0] == 0

        state, _, _, _ = env.step(0)
        assert state[0] == 38

        state = env.reset()
        assert state[0] == 0   

    def test_global_position(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        assert env.get_global_position((9, 9), (10, 10)) == get_global_position((9, 9), (10, 10))

    def test_local_position(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.reset()
        end = env.get_global_position((9, 9), (10, 10))
        assert [9, 9] == env.get_local_position(end)

    def test_solve_shortest(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.load("./maze/environment/utils/test/structure_test.txt")
        solve = env.solve()
        with open("./maze/environment/utils/test/solve_test.txt") as f:
            for line in f:
                test_solve = line

        assert solve == ast.literal_eval(test_solve)

    def test_change_start_and_goal(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.load("./maze/environment/utils/test/structure_test.txt")
        env.change_start_and_goal()
        assert env.start != (0, 0)
        
    def test_agent_random_position(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=False)
        env.load("./maze/environment/utils/test/structure_test.txt")
        agent_initial_position = env.agent
        env.agent_random_position()
        assert agent_initial_position != env.agent

    def test_size(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(5, 5), occlusion=False)
        state = env.reset()

        maze_size = (5, 5)
        maze_size = ((maze_size[0] * 2) - 1) ** 2
        assert maze_size + 3 == state.shape[0]

    def test_occlusion(self):
        TestCases.env = env = gym.make("Maze-v0", shape=(10, 10), occlusion=True)
        env.load("./maze/environment/utils/test/structure_test.txt")
        state = env.render("rgb_array")
        test_state = np.array(Image.open("./maze/environment/utils/test/occlusion_test.png"))
        assert (state == test_state).all()

if __name__ == "__main__":
    unittest.main()