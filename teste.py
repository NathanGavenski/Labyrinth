import time
import os
from os import listdir
from os.path import isfile, join

import gym
from maze import environment
import numpy as np
from PIL import Image
import torch
import signatory
from tqdm import tqdm

from maze.generate import generate


def state_to_action(source: int, target: int, shape: tuple) -> int:
    w, h = shape

    if source // w == target // w:
        if target > source:
            return 1
        else:
            return 3
    else:
        if target > source:
            return 0
        else:
            return 2


def list_mazes(path: str, folder: str) -> list:
    mypath = f'{path}{folder}/'
    return  [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

def measure_maze(path, size, bias=False):
    if not os.path.exists(path):
        os.makedirs(path)

    for t in ['train', 'eval']:
        mazes = list_mazes(f'./maze/environment/mazes/mazes{size[0]}/', t)

        trajectories = torch.Tensor(size=[0, 30])

        pbar = tqdm(mazes)
        pbar.set_description_str(f'Maze-v0 [Size:{size} Type:{t} Bias: {not bias}]')
        for maze in pbar:
            env = gym.make('Maze-v0', shape=size)
            env.load(maze)

            if bias and t == 'train':
                env.change_start_and_goal()

            solutions = [env.solve(mode='shortest')]
            for solution in solutions:
                env.reset(agent=True)

                episode = []
                for idx, tile in enumerate(solution[:-1]):
                    episode.append(env.agent)
                    action = state_to_action(tile, solution[idx + 1], size)
                    env.step(action)
                episode.append(env.agent)

                episode = np.array(episode)[None]
                episode = signatory.signature(torch.from_numpy(episode.astype(float)), 4)
                trajectories = torch.cat([trajectories, episode], dim=0)
                env.close()
            torch.save(trajectories, f'{path}{t}.pt')

    train = torch.load(f'{path}train.pt')
    valid = torch.load(f'{path}eval.pt')

    return round((valid * train).sum().item(), 4), round(torch.pow(train - valid, 2).sum().item(), 4)


def maze():
    if not os.path.exists('./tmp/experiments/'):
        os.makedirs('./tmp/experiments/')

    if not os.path.exists('./tmp/experiments/bias_measure.txt'):
        with open('./tmp/experiments/bias_measure.txt', 'w') as f:
            f.write('size;normal dot product;unbiased dot product;normal euclidean;unbiased euclidean\n')
    else:
        sizes = []
        with open('./tmp/experiments/bias_measure.txt') as f:
            for idx, line in enumerate(f):
                if idx > 0:
                    words = line.split(';')
                    sizes.append(eval(words[0]))

    all_sizes = [(5, 5), (10, 10), (25, 25), (50, 50), (100, 100)]
    for size in sizes:
        if size in all_sizes:
            all_sizes.remove(size)

    for size in all_sizes:
        if not os.path.exists(f'./maze/environment/mazes/mazes{size[0]}'):
            generate(100, 100, size, verbose=True)

        normal_dot, normal_euclidean = measure_maze('./tmp/experiments/', size)
        unbiased_dot, unbiased_euclidean = measure_maze('./tmp/experiments/', size, True)

        with open('./tmp/experiments/bias_measure.txt', 'a') as f:
            f.write(f'{size};{normal_dot};{unbiased_dot};{normal_euclidean};{unbiased_euclidean}\n')


def others():
    if not os.path.exists('./tmp/experiments/'):
        os.makedirs('./tmp/experiments/')

    if not os.path.exists('./tmp/experiments/bias_measure_others.txt'):
        with open('./tmp/experiments/bias_measure_others.txt', 'w') as f:
            f.write('name;size;dot product;euclidean\n')
    else:
        envs = []
        with open('./tmp/experiments/bias_measure_others.txt') as f:
            for idx, line in enumerate(f):
                if idx > 0:
                    words = line.split(';')
                    envs.append(eval(words[0]))

    path = './tmp/experts/'
    files = [f for f in listdir(path) if isfile(join(path, f)) and '.npz' in f]

    for env in files:
        if env in files:
            files.remove(env)
    
    for f in files:
        dataset = np.load(f'{path}{f}', allow_pickle=True)
        episode_starts = dataset['episode_starts']
        starts = np.where(episode_starts == True)[0]
        ends = starts - 1
        ends = np.append(ends[1:], episode_starts.shape[0] - 1)

        idx = starts.shape[0] // 2
        train = zip(starts[:idx], ends[:idx])
        valid = zip(starts[idx:], ends[idx:])

        obs = dataset['obs']   

        pbar = tqdm(range(starts.shape[0]))
        pbar.set_description_str(f'{f}')     

        for data, t in zip([train, valid], ['train', 'eval']):
            episode = obs[:ends[0]+1].astype(float)
            episode = torch.from_numpy(episode).unsqueeze(0)
            shape = signatory.signature(episode, 4).shape[-1]
            trajectories = torch.Tensor(size=[0, shape])
            torch.save(trajectories, f'{path}{t}.pt')
            
            for start, end in data:
                episode = obs[:end+1]
                episode = signatory.signature(torch.from_numpy(episode.astype(float)).unsqueeze(0).cuda(), 4)

                trajectories = torch.load(f'{path}{t}.pt')
                trajectories = torch.cat((trajectories, episode), dim=0)
                torch.save(trajectories, f'{path}{t}.pt')
                del trajectories

                pbar.update(1)
    
        train = torch.load(f'{path}train.pt')
        valid = torch.load(f'{path}eval.pt')

        dot = round((valid * train).sum().item(), 4)
        euclidean = round(torch.pow(train - valid, 2).sum().item(), 4)

        with open('./tmp/experiments/bias_measure_others.txt', 'a') as _f:
            _f.write(f'{f};{idx};{dot};{euclidean}')

if __name__ == '__main__':
    others()