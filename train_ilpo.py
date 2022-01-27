import os
from os import path
import shutil

import gym
import tensorflow as tf

from algo.il import ImageILPO, PolicyILPO, create_ilpo_dataset
from maze import environment

if __name__ == '__main__':
    if not path.exists('./dataset/ilpo_dataset'):
        create_ilpo_dataset(
            path='./dataset/dataset5',
            file='dataset.npy',
            output_dir='./dataset/ilpo_dataset',
        )

    #if os.path.exists('./tmp/ilpo/output/'):
    #    shutil.rmtree('./tmp/ilpo/output/')

    #tf.reset_default_graph()
    #ilpo = ImageILPO(
    #    input_dir='./dataset/ilpo_dataset',
    #    output_dir='./tmp/ilpo/output',
    #    checkpoint_dir='./tmp/ilpo/output',
    #    max_steps=None,
    #    max_epochs=5,
    #    batch_size=32,
    #    ngf=15,
    #)
    #ilpo.run()
    #exit()

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
            game=gym.make('Maze-v0', shape=(5, 5)),
            maze_path='./maze/environment/mazes/mazes5/',
            ngf=15,
            verbose=False,
            experiment=True,
            use_encoding=True,
            name='test'
        )

        policy.run_policy(times=10)
