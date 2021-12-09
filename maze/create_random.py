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
        '--amount',
        type=int
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

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    image_idx = 0
    dataset = np.ndarray(shape=[0, 4])
    amount_per_maze = (args.amount + 1*len(mazes))// len(mazes)
    for maze_idx, maze in enumerate(tqdm(mazes)):
        env = gym.make('Maze-v0', shape=(args.width, args.height))
        env.reset()
        env.load(maze)

        done = False
        for idx in range(amount_per_maze):
            image = env.render('rgb_array')
            np.save(f'{args.save_path}/{image_idx}', image)
            image_idx += 1
            
            if idx < amount_per_maze - 1:
                action = np.random.randint(0, 4)
                next_state, reward, done, info = env.step(action)
                entry = [maze_idx,image_idx,action,image_idx+1]  
                dataset = np.append(dataset, np.array(entry)[None], axis=0)  

                if done:
                    env.reset(agent=True)
        env.close()
    np.save(f'{args.save_path}/dataset', dataset)
