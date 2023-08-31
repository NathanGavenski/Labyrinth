"""Create heatmap of visited tiles by the expert."""
import argparse
from collections import defaultdict
from os import listdir, makedirs
from os.path import isfile, join, exists
from typing import List, Tuple

import gym
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
sns.set_theme()

# pylint: disable=[W0611, C0413]
from . import maze


def get_args() -> argparse.Namespace:
    """Get arguments from command line.

    Returns:
        argparse.Namespace: Arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Args for ploting expert ")

    # General
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether or not it should show the progress bar when creating mazes'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Whether or not it should save the plot'
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


def global_position_to_local(
        position: int,
        shape: Tuple[int, int] = (10, 10)
    ) -> Tuple[int, int]:
    """Converts a global position (0) to a local position (0, 0).

    Args:
        position (int): global position.
        shape (Tuple[int, int], optional): Maze width and height. Defaults to (10, 10).

    Returns:
        local (Tuple[int, int]): local position. 
    """
    return position // shape[0], position - (position // shape[0] * shape[0])


def percentage_to_rgb(minimum: int, maximum: int, value: int) -> Tuple[int, int, int]:
    """Converts a percentage to a rgb color.

    Args:
        minimum (int): Minimum value.
        maximum (int): Maximum value.
        value (int): Value to convert.

    Returns:
        RGB (Tuple[int, int, int]): RGB color.
    """
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value - minimum) / (maximum - minimum)
    blue = int(max(0, 255 * (1 - ratio)))
    red = int(max(0, 255 * (ratio - 1)))
    green = 255 - blue - red
    return red, green, blue


def plot_heatmap(data: List[int], name: str, axis: plt.axes) -> plt.axes:
    """Plot heatmap.

    Args:
        data (List[int]): Data of visited tiles by the expert.
        name (str): Name of the plot.
        ax (plt.axes): Axes of the plot.

    Returns:
        plt.axes: Axes of the plot.
    """
    axis = sns.heatmap(
        data[::-1],
        annot=False,
        fmt=".0%",
        annot_kws={"size": 10},
        linewidths=.5,
        ax=axis
    )
    axis.set_yticks([])
    axis.set_xticks([])
    axis.set_title(name)
    return axis

def plot_expert(args: argparse.Namespace) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create heatmap of visited tiles by the expert.

    Args:
        args (argparse.Namespace): Arguments from command line.

    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: Figure and axes of the plot.
    """
    diff = {}
    fig = plt.figure(figsize=(20, 5))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    axis = [ax1, ax2, ax3]

    for _type, axis in zip(['train', 'eval'], axis):
        mypath = f'{args.path}/{_type}/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath,f))]

        visited_tiles = defaultdict(int)
        for i in range((args.width * args.height)):
            visited_tiles[i] = 0

        max_visited = 0
        mazes = mazes if not args.verbose else tqdm(mazes)
        for _maze in mazes:
            env = gym.make('MazeScripts-v0', shape=(args.width, args.height))
            env.reset()
            env.load(_maze)
            solutions = env.solve(mode='all')
            max_visited += len(solutions)
            for solution in solutions:
                for tile in solution:
                    visited_tiles[tile] += 1

            env.close()

        data = np.zeros((args.width, args.height))
        for tile, times in visited_tiles.items():
            x_position, y_position = global_position_to_local(tile, shape=(args.width, args.height))
            data[x_position, y_position] = times / max_visited
        diff[_type] = data
        axis = plot_heatmap(data, _type, axis)

    heatmap_diff = np.absolute(diff['train'] - diff['eval'])
    axis[-1] = plot_heatmap(heatmap_diff, 'diff', axis[-1])
    return fig, axis


if __name__ == '__main__':
    arguments = get_args()
    plot_expert(arguments)

    plt.tight_layout()
    if arguments.save:
        if not exists('./plots/'):
            makedirs('./plots/')
        plt.savefig(f'./plots/heatmap{arguments.width}.png')
    else:
        plt.show()
