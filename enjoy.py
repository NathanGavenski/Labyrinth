import time
import os
from os import listdir
from os.path import isfile, join

import gym
import environment

def generate(split=.5, shape=(10, 10)):
    env = gym.make('Maze-v0', shape=shape)
    env.generate(amount=200)

    mypath = f'./environment/mazes/mazes{shape[0]}/'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    split_idx = int(len(files) * split)
    train = files[:split_idx]
    validation = files[split_idx:]
    print(f'Train: {len(train)}, Eval: {len(validation)}')

    os.makedirs(f'{mypath}train/')
    for f in train:
        os.rename(
            f'{mypath}{f}',
            f'{mypath}train/{f}'
        )
    os.makedirs(f'{mypath}eval/')
    for f in validation:
        os.rename(
            f'{mypath}{f}',
            f'{mypath}eval/{f}'
        )

if __name__ == "__main__":
    pass