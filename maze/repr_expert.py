import argparse
from collections import defaultdict
from os import listdir
from os.path import isfile, join

import gym
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import seaborn as sns
sns.set_theme()

import environment

def get_args():
    parser = argparse.ArgumentParser(
        description="Args for ploting expert "
    )

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

def global_position_to_local(position, shape=(10, 10)):
    y = position // shape[0]
    x = position - (y * shape[0])
    return (x, y)


def percentage_to_rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def plot_heatmap(data, name, ax):
    ax = sns.heatmap(
        data[::-1],
        annot=False,
        fmt=".0%",
        annot_kws={"size": 10},
        linewidths=.5,
        ax=ax
    )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(name)


if __name__ == '__main__':
    args = get_args()

    diff = {}
    
    fig = plt.figure(figsize=(20, 5))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    axis = [ax1, ax2, ax3]

    for _type, ax in zip(['train', 'eval'], axis):
        mypath = f'{args.path}/{_type}/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

        visited_tiles = defaultdict(int)
        for i in range((args.width * args.height)):
            visited_tiles[i] = 0

        mazes = mazes if not args.verbose else tqdm(mazes)
        for maze in mazes:
            env = gym.make('Maze-v0', shape=(args.width, args.height))
            env.reset()
            env.load(maze)
            solutions = env.solve(mode='all')
            for solution in solutions:
                for tile in solution:
                    visited_tiles[tile] += 1

            env.close()

        data = np.zeros((args.width, args.height))
        for tile, times in visited_tiles.items():
            x, y = global_position_to_local(tile, shape=(args.width, args.height))
            data[x, y] = (times / len(mazes))
        diff[_type] = data
        plot_heatmap(data, _type, ax)    

    heatmap_diff = np.absolute(diff['train'] - diff['eval'])
    plot_heatmap(heatmap_diff, 'diff', axis[-1])
    plt.tight_layout()
    if args.save:
        plt.savefig(f'./plots/heatmap.png')
    else:
        plt.show()
