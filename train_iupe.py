import argparse
import os

import gym

from maze import environment
from algo.il import IUPE

def get_args():
    parser = argparse.ArgumentParser(
        description="Args for creating expert dataset."
    )

    # General
    parser.add_argument(
        '--times',
        type=int,
        default=10,
        help='How many times should repeat each maze when unbiased is turned on'
    )

    parser.add_argument(
        '--idx',
        type=int,
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default='-1'
    )
    
    # Maze specific
    parser.add_argument(
        '--size',
        type=int,
        default=10,
        help="Width of the generated maze"
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    env = gym.make('Maze-v0', shape=(args.size, args.size))
    algo = IUPE(
        environment=env,
        maze_path=f'./maze/environment/mazes/mazes{args.size}/',
        width=args.size,
        height=args.size,
        episode_times=args.times,
        random_dataset=f'./dataset/random_dataset{args.size}/',
        expert_dataset=f'./dataset/dataset{args.size}/',
        device='cuda',
        batch_size=12,
        verbose=True,
        early_stop=True,
        name=f'{args.size}x{args.size}-{args.idx}'
    )
    algo.learn(100)
