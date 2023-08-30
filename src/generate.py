
import argparse
import os
from os import listdir
from os.path import isfile, join
import pathlib

import gym
from tqdm import tqdm

import maze


def get_args():
    parser = argparse.ArgumentParser(
        description="Args for generating multiple mazes."
    )

    # General
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Wheter or not it should show the progress bar when creating mazes'
    )
    parser.add_argument(
        '--train',
        type=int,
        default=100,
        help="How many mazes for the train set"
    )
    parser.add_argument(
        '--eval',
        type=int,
        default=100,
        help="How many mazes for the evaluation set"
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


def generate(train, eval, shape, verbose=False):
    env = gym.make('MazeScripts-v0', shape=shape)

    global_path = pathlib.Path(__file__).parent.resolve()
    mypath = f'{global_path}/environment/mazes/mazes{shape[0]}/'
    env.generate(mypath, amount=train+eval)
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    train_data = files[:train]
    evaluation_data = files[train:]
    print(f'Train: {len(train_data)}, Eval: {len(evaluation_data)}')

    os.makedirs(f'{mypath}train/')
    train_data = train_data if not verbose else tqdm(train_data)
    for f in train_data:
        os.rename(
            f'{mypath}{f}',
            f'{mypath}train/{f}'
        )
    os.makedirs(f'{mypath}eval/')
    evaluation_data = evaluation_data if not verbose else tqdm(evaluation_data)
    for f in evaluation_data:
        os.rename(
            f'{mypath}{f}',
            f'{mypath}eval/{f}'
        )


if __name__ == '__main__':
    args = get_args()

    generate(
        train=args.train,
        eval=args.eval,
        shape=(args.width, args.height)
    )
