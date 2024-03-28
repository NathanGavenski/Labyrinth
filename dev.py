import ast
from PIL import Image
import gym
from src import maze
from src.maze.utils.render import RenderUtils
import numpy as np

import logging
from src.create_maze_from_file import convert_from_file

logging.basicConfig(level=logging.ERROR)


# FIXME There is a bug where if you call step without rendering the environment
# first, the environment will not update the view correctly.
# This has to be a problem in the render function


def draw_maze(shape, lab):
    start = (0, 0)
    end = (shape[0] - 1, shape[1] - 1)

    render = RenderUtils(shape, screen_info=(800, 800)) \
        .draw_start(start) \
        .draw_end(end) \
        .draw_agent(start) \
        .draw_walls(lab)
    return render


def save_images(maze1, maze2, maze3) -> None:
    border = np.ones((maze1.shape[0], 1, 3)).astype("uint8")
    mazes = np.hstack((maze1, border, maze2, border, maze3))
    Image.fromarray(mazes).save("test.png")


def sort_visited(visited):
    return list(set(map(tuple, sorted(map(sorted, visited)))))


if __name__ == "__main__":
    env = gym.make("Maze-v0", shape=(10, 10))
    env.reset()
    env.save("./tests/tmp.txt")
    first_solutions = env.solve("all")
    first_solutions.sort(key=len)
    env.close()

    env = gym.make("Maze-v0", shape=(10, 10))
    env.load("./tests/tmp.txt")
    env.reset()
    second_solutions = env.solve("all")
    second_solutions.sort(key=len)

    for _x, _y in zip(first_solutions, second_solutions):
        assert len(_x) == len(_y)
        assert (set(_y) - set(_x)) == set()


    # env = gym.make("Maze-v0", shape=(5, 5))
    # env.load("test.txt")

    # solutions = env.solve(mode="all")
    # print(len(solutions))
    # env.render("rgb_array")
    # print()
    # state = env.reset()
    # floors = []
    # for floor in env.ice_floors:
    #     floor = env.translate_position(floor)
    #     floor = env.get_global_position(floor, env.maze[1:-1, 1:-1].shape)
    #     floors.append(floor)
    # Image.fromarray(env.render("rgb_array")).save("test.png")
    # exit()

    # assert (state[3:len(floors) + 3] == floors).all()

    # # Validate trajectory
    # ice_floor = [env.get_global_position(floor) for floor in env.ice_floors]
    # print(ice_floor, env.dfs.start)

    # path = env.dfs.find_path(
    #     env.undirected_pathways,
    #     env.dfs.start,
    #     ice_floor[0],
    #     early_stop=True
    # )
    # path = [node.identifier for node in path[ice_floor[0]].d]
    # intersec = list(set(path).intersection(set(ice_floor)))
    # indexes = [np.where(path == floor) for floor in intersec]
    # for i in [5, 10, 25, 50, 100, 250]:
    #     shape = (i, i)
    #     env = gym.make(
    #         "Maze-v0",
    #         shape=shape,
    #         occlusion=False,
    #         key_and_door=True,
    #         icy_floor=False
    #     )
    #     print(f"Created maze: ({i}, {i})")
    #     env.reset()
    #     Image.fromarray(env.render("rgb_array")).save("test.png")
    #     env.save("test.txt")
    #     env.close()
    #     del env
    #     exit()
