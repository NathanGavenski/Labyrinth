"""Create maze structures for maze environment."""
import argparse
import os
from os import listdir
from os.path import isfile, join
import pathlib
from typing import Optional, Tuple

import gym
from tqdm import tqdm

# pylint: disable=[W0611]
import maze

import logging
logging.basicConfig(level=logging.ERROR)

def get_args() -> argparse.Namespace:
    """Get arguments from command line.

    Returns:
        argparse.Namespace: arguments from command line
    """
    parser = argparse.ArgumentParser(description="Args for generating multiple mazes.")

    # General
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether or not it should show the progress bar when creating mazes'
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


def generate(
        train_amount: int,
        eval_amount: int,
        shape: Tuple[int, int],
        verbose: Optional[bool] = False
    ) -> None:
    """_summary_

    Args:
        train (int): Amount of mazes for the train set
        eval (int): Amount of mazes for the evaluation set
        shape (Tuple[int, int]): Width and height of the generated mazes
        verbose (Optional[bool]): Whether it should show the progress bar when creating mazes. 
            Defaults to False.
    """
    env = gym.make('MazeScripts-v0', shape=shape)

    global_path = pathlib.Path(__file__).parent.resolve()
    mypath = f'{global_path}/environment/mazes/mazes{shape[0]}/'
    env.generate(mypath, amount=train_amount + eval_amount)
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    train_data = files[:train_amount]
    evaluation_data = files[train_amount:]
    print(f'Train: {len(train_data)}, Eval: {len(evaluation_data)}')

    os.makedirs(f'{mypath}train/')
    train_data = train_data if not verbose else tqdm(train_data)
    for _file in train_data:
        os.rename(
            f'{mypath}{_file}',
            f'{mypath}train/{_file}'
        )
    os.makedirs(f'{mypath}eval/')
    evaluation_data = evaluation_data if not verbose else tqdm(evaluation_data)
    for _file in evaluation_data:
        os.rename(f'{mypath}{_file}', f'{mypath}eval/{_file}')


if __name__ == '__main__':
    args = get_args()

    generate(
        train_amount=args.train,
        eval_amount=args.eval,
        shape=(args.width, args.height),
        verbose=args.verbose
    )
