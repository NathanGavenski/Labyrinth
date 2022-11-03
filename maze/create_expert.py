import argparse
import os
from os import listdir
from os.path import isfile, join
import shutil

import gym
import numpy as np
from PIL import Image
from tqdm import tqdm

import environment


def get_args():
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


def state_to_action(source: int, target: int, shape: tuple) -> int:
    w, h = shape

    # Test left or right
    if source // w == target // w:
        if target > source:
            return 1
        else:
            return 3
    else:
        if target > source:
            return 0
        else:
            return 2


if __name__ == '__main__':

    args = get_args()

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
    for maze_idx, maze in enumerate(tqdm(mazes)):
        env = gym.make('MazeScripts-v0', shape=(args.width, args.height))
        env.load(maze)
        
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
                        solution[idx+1],
                        shape=(args.width, args.height)
                    )
                    state, reward, done, info = env.step(action)
                    total_reward += reward

                if not done:
                    entry = [
                        maze_idx,  # maze version
                        solution_idx,  # solution number
                        image_idx-1,  # state 
                        action,  # action
                        image_idx,  # next_state
                        0,  # episode reward
                        reward,  # step reward
                        True if idx == 0 else False,  # episode_starts
                        False,  # episode_ends                        
                    ]
                    dataset = np.append(
                        dataset,
                        np.array(entry)[None],
                        axis=0
                    )
                elif done:
                    dataset[-1, 5] = total_reward
                    dataset[-1, -1] = True

            env.close()
    np.save(f'{args.save_path}/dataset', dataset)
