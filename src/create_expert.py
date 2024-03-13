"""Create teacher dataset for imitation learning."""
import argparse
import os
from os import listdir
from os.path import isfile, join
import shutil
from typing import Any, List, Tuple

import gym
import numpy as np
from tqdm import tqdm

from . import maze
import logging
logging.basicConfig(level=logging.ERROR)


def get_args() -> argparse.Namespace:
    """Get arguments from command line.

    Returns:
        argparse.Namespace: arguments from command line
    """
    parser = argparse.ArgumentParser(
        description="Args for creating expert dataset."
    )

    # General
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether or not it should show the progress bar when creating the dataset'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        help='Where should the function save the dataset'
    )
    parser.add_argument(
        '--path',
        type=str
    )
    parser.add_argument(
        '--unbiased',
        action='store_true',
        help='Swaps start and goal for each maze in order to reduce bias'
    )
    parser.add_argument(
        '--times',
        type=int,
        default=10,
        help='How many times should repeat each maze when unbiased is turned on'
    )

    # Maze specific
    parser.add_argument(
        '--width',
        type=int,
        default=10,
        help="Width of the generated maze"
    )
    parser.add_argument(
        '--height',
        type=int,
        default=10,
        help="Height of the generated maze"
    )

    return parser.parse_args()


def state_to_action(source: int, target: int, shape: Tuple[int, int]) -> int:
    """Convert global index states into action (UP, DOWN, LEFT and RIGHT).

    Args:
        source (int): global index for the source in the maze.
        target (int): global index for the target in the maze.
        shape (Tuple[int, int]): maze shape.

    Returns:
        int: action to take.
    """
    width, _ = shape

    # Test left or right
    if source // width == target // width:
        if target > source:
            return 1
        return 3

    if target > source:
        return 0
    return 2


def create(args: argparse.Namespace) -> List[Any]:
    """Create expert dataset for imitation learning.

    Args:
        args (argparse.Namespace): Arguments from command line

    Returns:
        List[Any]: Expert dataset for Imitation Learning.
        TODO explain what is in the dataset.
    """
    mypath = f'{args.path}/train/'
    mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    if args.unbiased:
        mazes = np.repeat(mazes, args.times, axis=0)

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    image_idx = 0
    dataset = np.ndarray(shape=[0, 9])
    for maze_idx, _maze in enumerate(tqdm(mazes)):
        env = gym.make(
            'MazeScripts-v0',
            shape=(args.width, args.height),
            screen_width=600,
            screen_height=600
        )
        env.load(_maze)

        if args.unbiased and (maze_idx % args.times != 0):
            env.change_start_and_goal()

        solutions = env.solve(mode='all')
        for solution_idx, solution in enumerate(solutions):
            env.reset(agent=True)
            done = False

            total_reward = 0
            for idx, tile in enumerate(solution):
                image = env.render('rgb_array')
                np.save(f'{args.save_path}/{image_idx}', image)
                image_idx += 1

                if idx < len(solution) - 1:
                    action = state_to_action(
                        tile,
                        solution[idx + 1],
                        shape=(args.width, args.height)
                    )
                    try:
                        _, reward, done, _ = env.step(action)
                    except Exception as e:
                        print(e)
                        print(_maze)
                        exit()
                    total_reward += reward

                    entry = [
                        maze_idx,  # maze version
                        solution_idx,  # solution number
                        image_idx - 1,  # state
                        action,  # action
                        image_idx,  # next_state
                        0,  # episode reward
                        reward,  # step reward
                        idx == 0,  # episode_starts
                        False,  # episode_ends
                    ]
                    dataset = np.append(dataset, np.array(entry)[None], axis=0)
                if done:
                    image = env.render('rgb_array')
                    np.save(f'{args.save_path}/{image_idx}', image)
                    image_idx += 1
                    dataset[-1, 5] = total_reward
                    dataset[-1, -1] = True

            env.close()
        del env
    return dataset


if __name__ == '__main__':
    arguments = get_args()
    expert_dataset = create(arguments)
    np.save(f'{arguments.save_path}/dataset', expert_dataset)
