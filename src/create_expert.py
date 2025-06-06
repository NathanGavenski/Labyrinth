"""Create teacher dataset for imitation learning."""
import argparse
import os
from os import listdir
from os.path import isfile, join
import shutil
from typing import Any, List, Tuple
import logging

import gymnasium as gym
import numpy as np
from tqdm import tqdm

try:
    from . import labyrinth
    from .labyrinth import file_utils
except ImportError:
    import labyrinth
    from labyrinth import file_utils

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
        help='Swaps start and goal for each labyrinth in order to reduce bias'
    )
    parser.add_argument(
        '--times',
        type=int,
        default=10,
        help='How many times should repeat each labyrinth when unbiased is turned on'
    )
    parser.add_argument(
        '--folder',
        type=str,
        default='train',
        help='Which folder to use from path'
    )

    # labyrinth specific
    parser.add_argument(
        '--width',
        type=int,
        default=10,
        help="Width of the generated labyrinth"
    )
    parser.add_argument(
        '--height',
        type=int,
        default=10,
        help="Height of the generated labyrinth"
    )
    parser.add_argument(
        '--shortest',
        action='store_true',
        help='If it should only use the shortest solution for each labyrinth'
    )

    return parser.parse_args()


def state_to_action(source: int, target: int, shape: Tuple[int, int]) -> int:
    """Convert global index states into action (UP, DOWN, LEFT and RIGHT).

    Args:
        source (int): global index for the source in the labyrinth.
        target (int): global index for the target in the labyrinth.
        shape (Tuple[int, int]): labyrinth shape.

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


def create(args: argparse.Namespace, folder: str = 'train') -> List[Any]:
    """Create expert dataset for imitation learning.

    Args:
        args (argparse.Namespace): Arguments from command line

    Returns:
        List[Any]: Expert dataset for Imitation Learning.
        TODO explain what is in the dataset.
    """
    mypath = f'{args.path}/{folder}/'
    labyrinths = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    if args.unbiased:
        labyrinths = np.repeat(labyrinths, args.times, axis=0)

    save_path = f'{args.save_path}{folder}/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_idx = 0
    dataset = np.ndarray(shape=[0, 10])
    for lab_idx, labyrinth in enumerate(tqdm(labyrinths)):
        env = gym.make(
            'Labyrinth-v0',
            shape=(args.width, args.height),
            screen_width=600,
            screen_height=600,
            render_mode="rgb_array"
        )
        env.load(*file_utils.convert_from_file(labyrinth))

        if args.unbiased and (lab_idx % args.times != 0):
            env.change_start_and_goal()

        solutions = env.solve(mode='all' if not args.shortest else 'shortest')
        for solution_idx, solution in enumerate(solutions):
            env.reset(options={"agent": True})
            done = False

            total_reward = 0
            for idx, tile in enumerate(solution):
                image = env.render()
                np.save(f'{save_path}/{image_idx}', image)
                image_idx += 1

                if idx < len(solution) - 1:
                    action = state_to_action(
                        tile,
                        solution[idx + 1],
                        shape=(args.width, args.height)
                    )
                    _, reward, done, terminated, _ = env.step(action)
                    done |= terminated
                    total_reward += reward

                    entry = [
                        lab_idx,  # labyrinth version
                        solution_idx,  # solution number
                        image_idx - 1,  # state
                        action,  # action
                        image_idx,  # next_state
                        0 if not done else total_reward,  # episode reward
                        reward,  # step reward
                        idx == 0,  # episode_starts
                        done,  # episode_ends
                        labyrinth.split('/')[-1]  # labyrinth name
                    ]
                    dataset = np.append(dataset, np.array(entry)[None], axis=0)
            if done:
                image = env.render()
                np.save(f'{save_path}/{image_idx}', image)
                image_idx += 1

            env.close()
        del env
    return dataset


if __name__ == '__main__':
    arguments = get_args()
    folders = arguments.folder.split(',')
    for _folder in folders:
        expert_dataset = create(arguments, _folder)
        np.save(f'{arguments.save_path}{_folder}/dataset', expert_dataset)
