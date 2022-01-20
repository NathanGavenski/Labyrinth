from os import path
from algo.il import ImageILPO, PolicyILPO, create_ilpo_dataset

import gym
import tensorflow as tf

from maze import environment

if __name__ == '__main__':
    if not path.exists('./dataset/ilpo_dataset'):
        create_ilpo_dataset(
            path='./dataset/dataset5',
            file='dataset.npy',
            output_dir='./dataset/ilpo_dataset',
        )

    ilpo = ImageILPO(
        input_dir='./dataset/ilpo_dataset',
        output_dir='./tmp/ilpo/output',
        checkpoint_dir='./tmp/ilpo/checkpoint',
        batch_size=1,
        ngf=15,
    )
    ilpo.run()

    # game = gym.make('Maze-v0', shape=(5, 5))

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    
    # with sess.as_default():
    #     policy = PolicyILPO(
    #         sess,
    #         shape=[None, 128, 128, 3],
    #         checkpoint='./tmp/ilpo/checkpoint',
    #         game=game,
    #         maze_path='./maze/environment/mazes/mazes5/',
    #         ngf=15,
    #         verbose=False,
    #         experiment=True,
    #         use_encoding=True,
    #         # save_path='./tmp/ilpot/output',
    #     )

    #     policy.run_policy(times=1)
