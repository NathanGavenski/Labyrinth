from operator import length_hint
import os
from os import listdir
from os.path import isfile, join
from tabnanny import verbose

import numpy as np
from PIL import Image
from tqdm import tqdm


from imitation.algorithms.adversarial import gail
from imitation.data import types
import stable_baselines3 as sb3
from stable_baselines3.common import vec_env

class GAIL:
    def __init__(self, dataset, game, maze_path, folder='./tmp/gail/', batch_size=32, pretrain=2048) -> None:
        mypath = f'{maze_path}train/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.mazes = np.array(mazes)

        mypath = f'{maze_path}eval/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.eval_mazes = np.array(mazes)


        self.dataset = dataset

        self.original_game = game
        self.game = vec_env.DummyVecEnv([lambda: game])
        self.game = vec_env.VecTransposeImage(self.game)

        self.model = gail.GAIL(
            venv=self.game,
            demonstrations=self.dataset,
            demo_batch_size=batch_size,
            gen_algo=sb3.PPO('CnnPolicy', self.game, verbose=0, n_steps=pretrain),
            allow_variable_horizon=True
        )

        if not os.path.exists(folder):
            os.makedirs(folder)

    def run(self, total_timesteps=1e6):
        self.model.train(int(total_timesteps))

    # TODO transform this into an EvalCallback
    def eval(self, eval=True, soft=True):

        if eval:
            mazes = self.eval_mazes
        else:
            mazes = self.mazes

        ratio = []
        episode_reward = []
        for maze in mazes:
            total_reward, done = 0, False
            self.original_game.reset()
            self.original_game.load(maze)
            if soft:
                w, h = self.original_game.shape
                self.original_game.change_start_and_goal(min_distance=(w + h) // 2)
            
            while not done:
                obs = self.original_game.render('rgb_array')
                obs = np.transpose(obs, (2, 0, 1))
                action, _ = self.model.gen_algo.predict(obs.copy())
                obs, reward, done, info = self.original_game.step(action)
                total_reward += reward
            
            ratio.append((info['state'][:2] == self.original_game.end).all())
            episode_reward.append(total_reward)
        
        return np.mean(episode_reward), np.mean(ratio)


def create_dataset(path, file, times=10):
    original_dataset = np.load(f'{path}/{file}', allow_pickle=True)
    original_dataset = np.repeat(original_dataset, times, axis=0)

    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts = {key: [] for key in keys}
    for data in original_dataset:
        maze, solution, \
            state, action, next_state, \
            episode_reward, reward, \
            episode_start, episode_ends = data.astype(int)

        obs = np.load(f'{path}/{state}.npy', allow_pickle=True)
        obs = np.transpose(obs, (2, 0, 1))
        parts['obs'].append(obs)

        obs = np.load(f'{path}/{next_state}.npy', allow_pickle=True)
        obs = np.transpose(obs, (2, 0, 1))
        parts['next_obs'].append(obs)
        
        parts['acts'].append(action)
        parts['dones'].append(episode_ends)
        parts['infos'].append([{}])

    parts['obs'] = np.array(parts['obs'])
    parts['next_obs'] = np.array(parts['next_obs'])
    parts['acts'] = np.array(parts['acts'])
    parts['dones'] = np.array(parts['dones']).astype(bool)
    parts['infos'] = np.array(parts['infos'])

    lengths = set(map(len, parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.Transitions(**parts)
