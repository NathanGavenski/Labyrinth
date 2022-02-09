import argparse
import os

import gym

from maze import environment
from algo.il import GAIL, create_gail_dataset


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

    print('Creating dataset')
    path, file = f'./dataset/dataset{args.size}', 'dataset.npy'
    dataset = create_gail_dataset(path, file, times=args.times)

    print(f'Running {args.idx}')
    env = gym.make('Maze-v0', shape=(args.size, args.size))
    model = GAIL(
        dataset=dataset,
        game=env,
        maze_path=f'./maze/environment/mazes/mazes{args.size}/',
        log_name=f'{args.size}x{args.size}-{args.idx}'
    )
    model.run()
