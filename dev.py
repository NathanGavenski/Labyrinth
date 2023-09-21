from PIL import Image
import gym
from src import maze
from src.maze.utils.render import RenderUtils
import numpy as np


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
    shape = (5, 5)
    env = gym.make(
        "Maze-v0",
        shape=shape,
        occlusion=False,
        key_and_door=False,
        icy_floor=False
    )
    (maze_o, visited_o), (maze_e, visited_e) = env.set_ice_floors()
    env.reset()

    # What I need to do is to solve each maze and ensure that both paths are:
    # 1. Different from each other at some point
    #   1.1 Maybe have a min length of difference
    # 2. Are present in the new maze ✔️
    # This seems to be the best option so far

    visited_e = sort_visited(visited_e)
    visited_e.sort()

    env.maze = maze_o
    env._pathways = visited_o
    env.pathways = env.define_pathways(visited_o)
    solution_1 = env.solve("shortest")

    env.maze = maze_e
    env._pathways = visited_e
    env.pathways = env.define_pathways(visited_e)
    solution_2 = env.solve("shortest")

    solution_pairs = list(zip(solution_2, solution_2[1:] + solution_2[:1]))[:-1]
    solution_pairs = sort_visited(solution_pairs)
    solution_pairs.sort()

    print(visited_e)
    print(solution_pairs)
    assert set(solution_pairs).issubset(set(visited_e))

    new_visited = list(set(visited_o + solution_pairs))
    new_visited.sort()

    new_maze, new_visited = env._generate(new_visited)

    # Although it works, it creates a maze with too many solutions and I'm left with
    # finding the pathways in the new maze (inverse engineering it -- I did it lol)
    image_o = draw_maze(shape, maze_o).viewer.render(return_rgb_array=True)
    image_e = draw_maze(shape, maze_e).viewer.render(return_rgb_array=True)
    # new_maze_2 = np.logical_and(maze_o, maze_e).astype(float)

    # This is the reverse engineering process rofl
    # visited_o = sort_visited(visited_o)
    # visited_o.sort()
    # visited_e = sort_visited(visited_e)
    # visited_e.sort()

    # new_visited = list(set(visited_o + visited_e))
    # new_visited.sort()
    # new_maze, _ = env._generate(new_visited)
    # assert (new_maze == new_maze_2).all()

    image_new = draw_maze(shape, new_maze).viewer.render(return_rgb_array=True)
    save_images(image_o, image_e, image_new)

    env.maze = new_maze
    env._pathways = new_visited
    env.pathways = env.define_pathways(new_visited)
    solutions = env.solve("all")
    print(len(solutions))
    for solution in solutions:
        print(len(solution))
        print(solution)
    print(list(map(len, env.solve("all"))))
