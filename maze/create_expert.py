import argparse
import os
from os import listdir
from os.path import isfile, join
import shutil

import gym
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

def state_to_action(source:int, target:int, shape:tuple) -> int:
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

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(f'{args.save_path}/dataset.txt', 'w') as f:
        image_idx = 0

        for maze_idx, maze in tqdm(enumerate(mazes)):
            env = gym.make('Maze-v0', shape=(args.width, args.height))
            env.reset()
            env.load(maze)
            solutions = env.solve(mode='all')

            for solution_idx, solution in enumerate(solutions):
                env.reset(agent=True)
                for idx, tile in enumerate(solution):
                    image = env.render('rgb_array')
                    Image.fromarray(image).save(f'{args.save_path}/{image_idx}.png')
                    image_idx += 1

                    if idx < len(solution) - 1:
                        action = state_to_action(tile, solution[idx+1], shape=(args.width, args.height))
                        env.step(action)
                        f.write(f'{maze_idx};{solution_idx}{image_idx};{action};{image_idx+1}\n')               
            env.close()