"""Create labyrinth structures for labyrinth environment."""
import argparse
import os
from os import listdir
from os.path import isfile, join
import pathlib
from typing import Optional, Tuple
import logging

import gymnasium as gym
from tqdm import tqdm

try:
    from . import labyrinth
except ImportError:
    import labyrinth

logging.basicConfig(level=logging.ERROR)


def get_args() -> argparse.Namespace:
    """Get arguments from command line.

    Returns:
        argparse.Namespace: arguments from command line
    """
    parser = argparse.ArgumentParser(description="Args for generating multiple labyrinths.")

    # General
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether or not it should show the progress bar when creating labyrinths'
    )
    parser.add_argument(
        '--train',
        type=int,
        default=100,
        help="How many labyrinths for the train set"
    )
    parser.add_argument(
        '--eval',
        type=int,
        default=100,
        help="How many labyrinths for the evaluation set"
    )
    parser.add_argument(
        '--test',
        type=int,
        default=100,
        help="How many labyrinths for the test set"
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

    return parser.parse_args()


def generate(
    train_amount: int,
    eval_amount: int,
    test_amount: int,
    shape: Tuple[int, int],
    verbose: Optional[bool] = False
) -> None:
    """Generate labyrinths for the labyrinth environment.

    Args:
        train (int): Amount of labyrinths for the train set
        eval (int): Amount of labyrinths for the evaluation set
        shape (Tuple[int, int]): Width and height of the generated labyrinths
        verbose (Optional[bool]): Whether it should show the progress bar when
            creating labyrinths. Defaults to False.
    """
    env = gym.make('Labyrinth-v0', shape=shape)

    global_path = pathlib.Path(__file__).parent.resolve()
    mypath = f'{global_path}/environment/labyrinths/labyrinths{shape[0]}/'
    env.generate(mypath, amount=train_amount + eval_amount + test_amount, verbose=verbose)
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    eval_amount += train_amount

    train_data = files[:train_amount]
    evaluation_data = files[train_amount:eval_amount]
    test_data = files[eval_amount:]
    print(
        f'Train: {len(train_data)},',
        f'Eval: {len(evaluation_data)},',
        f'Test: {len(test_data)}'
    )

    os.makedirs(f'{mypath}train/', exist_ok=True)
    train_data = train_data if not verbose else tqdm(train_data)
    for _file in train_data:
        os.rename(
            f'{mypath}{_file}',
            f'{mypath}train/{_file}'
        )
    os.makedirs(f'{mypath}eval/', exist_ok=True)
    evaluation_data = evaluation_data if not verbose else tqdm(evaluation_data)
    for _file in evaluation_data:
        os.rename(f'{mypath}{_file}', f'{mypath}eval/{_file}')

    os.makedirs(f'{mypath}test/', exist_ok=True)
    test_data = test_data if not verbose else tqdm(test_data)
    for _file in test_data:
        os.rename(f'{mypath}{_file}', f'{mypath}test/{_file}')


if __name__ == '__main__':
    args = get_args()

    generate(
        train_amount=args.train,
        eval_amount=args.eval,
        test_amount=args.test,
        shape=(args.width, args.height),
        verbose=args.verbose
    )
