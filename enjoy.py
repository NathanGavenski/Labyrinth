import time
import os
from os import listdir
from os.path import isfile, join

import gym
import environment

from algos import DQN

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

def dqn(files=[]):
    import math
    
    import numpy as np
    from torch import optim
    from tqdm import tqdm

    from algos import DQN, ReplayBuffer, compute_td_loss

    env = gym.make('Maze-v0')
    model = DQN(env.reset().shape[0], env.action_space.n)
        
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(1000)

    batch_size = 32
    gamma      = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    frame_idx = 0
    episode_rewards = []
    for e in range(10000):

        losses = [] 
        all_rewards = []
        for maze in files:
            state, done = env.load(maze), False
            env.reset(agent=True)
            episode_reward = 0

            while not done:
                epsilon = epsilon_by_frame(frame_idx)
                action = model.act(state, epsilon)
                
                next_state, reward, done, _ = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                
                frame_idx += 1
                state = next_state
                episode_reward += reward

                if len(replay_buffer) > batch_size:
                    loss = compute_td_loss(batch_size, gamma, optimizer, model, replay_buffer)
                    losses.append(loss.data.item())

            all_rewards.append(episode_reward)
        episode_rewards.append(all_rewards)
        print(e, np.mean(all_rewards), np.mean(losses))
    
    return model, episode_rewards


if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join
    import numpy as np

    mypath = './environment/mazes/mazes10/train/'
    mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    _, rewards = dqn(mazes)
    rewards = np.array(rewards)
    
    print(rewards.shape)
    print(rewards.mean(axis=0))