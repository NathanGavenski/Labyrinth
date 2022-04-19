import argparse
import os
from os import listdir
from os.path import isfile, join
import shutil

import gym
import numpy as np
from tqdm import tqdm
from PIL import Image

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


if __name__ == '__main__':

    args = get_args()

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
    for maze_idx, maze in enumerate(mazes):
        env = gym.make('MazeScripts-v0', shape=(args.width, args.height))
        env.load(maze)

        if args.unbiased and (maze_idx % args.times != 0):
            env.change_start_and_goal()

        if args.random_start and (maze_idx % args.times != 0):
            env.agent_random_position()

        state = env.agent
        idx = 0
        done = False
        while idx < amount_per_maze - 1:
            image = env.render('rgb_array')
            action = np.random.randint(0, 4)
            _, reward, done, info = env.step(action) 
            next_state = env.agent

            if (state != next_state):
                # state
                np.save(f'{args.save_path}/{image_idx}', image)
                image_idx += 1

                # next state
                image = env.render('rgb_array')
                np.save(f'{args.save_path}/{image_idx}', image)
                image_idx += 1

                entry = [maze_idx, image_idx - 2, action, image_idx - 1]
                dataset = np.append(dataset, np.array(entry).astype(int)[None], axis=0)
                pbar.update()
                idx += 1

            if done:
                env.reset(agent=True)
                state = env.agent
            else:
                state = next_state

        env.close()

    np.save(f'{args.save_path}/dataset', dataset.astype(int))
