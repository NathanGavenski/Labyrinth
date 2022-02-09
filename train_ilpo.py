import argparse
import os
from os import path
import shutil

import gym
import tensorflow as tf

from algo.il import ImageILPO, PolicyILPO, create_ilpo_dataset
from maze import environment

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

    if not path.exists('./dataset/ilpo_dataset'):
        create_ilpo_dataset(
            path=f'./dataset/dataset{args.size}',
            file='dataset.npy',
            output_dir='./dataset/ilpo_dataset',
        )

    if os.path.exists('./tmp/ilpo/output/'):
       shutil.rmtree('./tmp/ilpo/output/')

    tf.reset_default_graph()
    ilpo = ImageILPO(
       input_dir='./dataset/ilpo_dataset',
       output_dir='./tmp/ilpo/output',
       checkpoint_dir='./tmp/ilpo/output',
       max_steps=None,
       max_epochs=5,
       batch_size=32,
       ngf=15,
    )
    ilpo.run()

    tf.reset_default_graph()
    config = tf.ConfigProto(
        inter_op_parallelism_threads=4,
        intra_op_parallelism_threads=4,
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
        
    with sess.as_default():
        policy = PolicyILPO(
            sess,
            shape=[None, 128, 128, 3],
            checkpoint='./tmp/ilpo/output/',
            game=gym.make('Maze-v0', shape=(args.size, args.size)),
            maze_path=f'./maze/environment/mazes/mazes{args.size}/',
            ngf=15,
            verbose=False,
            experiment=True,
            use_encoding=True,
            name=f'{args.size}x{args.size}-{args.idx}'
        )

        policy.run_policy(times=args.times)
