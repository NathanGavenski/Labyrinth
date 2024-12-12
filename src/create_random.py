"""Create random dataset for imitation learning."""
import argparse
import os
from os import listdir
from os.path import isfile, join
import shutil
from typing import Any, List

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from . import maze


def get_args() -> argparse.Namespace:
    """Get arguments from command line.

    Returns:
        arguments (Namespace): Arguments from command line.
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
        '--amount',
        type=int
    )
    parser.add_argument(
        '--unbiased',
        action='store_true',
        help='Swaps start and goal for each maze in order to reduce bias'
    )
    parser.add_argument(
        '--random_start',
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


def generate(args: argparse.Namespace) -> List[Any]:
    """Create random dataset for imitation learning.

    Args:
        args (argparse.Namespace): Arguments from command line.

    Returns:
        List[Any]: Random dataset for imitation learning. 
        TODO explain what is in the dataset.
    """
    mypath = f'{args.path}/train/'
    mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    if args.unbiased or args.random_start:
        mazes = np.repeat(mazes, args.times, axis=0)

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    image_idx = 0
    dataset = np.ndarray(shape=[0, 4])
    amount_per_maze = (args.amount + 1 * len(mazes)) // len(mazes)
    pbar = tqdm(range(args.amount))
    for maze_idx, _maze in enumerate(mazes):
        env = gym.make('Maze-v0', shape=(args.width, args.height), render_mode="rgb_array")
        env.load(_maze)

        if args.unbiased and (maze_idx % args.times != 0):
            env.change_start_and_goal()

        if args.random_start and (maze_idx % args.times != 0):
            env.agent_random_position()

        idx = 0
        done = False
        state = env.agent
        while idx < amount_per_maze - 1:
            image = env.render()
            action = np.random.randint(0, 4)
            _, _, done, terminated, _ = env.step(action)
            done |= terminated
            next_state = env.agent

            if state != next_state:
                # state
                np.save(f'{args.save_path}/{image_idx}', image)
                image_idx += 1

                # next state
                image = env.render()
                np.save(f'{args.save_path}/{image_idx}', image)
                image_idx += 1

                entry = [maze_idx, image_idx - 2, action, image_idx - 1]
                dataset = np.append(
                    dataset, np.array(entry).astype(int)[None], axis=0)
                pbar.update()
                idx += 1

            if done:
                env.reset(options={"agent": True})
                state = env.agent
            else:
                state = next_state

        env.close()
    return dataset.astype(int)


if __name__ == '__main__':
    arguments = get_args()
    random_dataset = generate(arguments)
    np.save(f'{arguments.save_path}/dataset', random_dataset)
