from ctypes import Union
import os
from os import listdir
from os.path import isfile, join
from typing import Optional, Union

import gym
from imitation.algorithms.adversarial import gail
from imitation.data import types
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common import vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback as StableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from tensorboard_wrapper.tensorboard import Tensorboard as Board

class EvalCallback(StableEvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        maze_path: str,
        log_name: str,
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 1000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        determionistic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ) -> None:
        super(EvalCallback, self).__init__(
            eval_env, 
            callback_on_new_best, 
            n_eval_episodes, 
            eval_freq,
            log_path,
            best_model_save_path,
            determionistic,
            render,
            verbose,
            warn
        )
        mypath = f'{maze_path}train/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.mazes = np.array(mazes)

        mypath = f'{maze_path}eval/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        self.eval_mazes = np.array(mazes)
        
        self.original_game = eval_env
        self.board = Board(f'GAIL-{log_name}', './tmp/board/', delete=True)

    def eval(self, eval: bool = True, soft: bool = False, occlusion: bool = False) -> tuple:

        if eval:
            mazes = self.eval_mazes
        else:
            mazes = self.mazes

        timesteps = 0
        ratio = []
        episode_reward = []
        for maze in mazes:
            total_reward, done = 0, False
            self.original_game.load(maze)
            
            if soft:
                w, h = self.original_game.shape
                self.original_game.change_start_and_goal(min_distance=(w + h) // 2)

            if occlusion:
                self.original_game.set_occlusion_on()
            else:
                self.original_game.set_occlusion_off()
            
            while not done:
                obs = self.original_game.render('rgb_array')
                obs = np.transpose(obs, (2, 0, 1))
                action, _ = self.model.gen_algo.predict(obs.copy())
                obs, reward, done, info = self.original_game.step(action)
                total_reward += reward
                timesteps += 1
            
            ratio.append((info['state'][:2] == self.original_game.end).all())
            episode_reward.append(total_reward)
        
        return np.mean(episode_reward), np.mean(ratio)

    def __call__(self) -> bool:
        print('Evaluating')
        # Eval (train)
        aer, ratio = self.eval(eval=False, soft=False)
        self.board.add_scalars(
            prior='Policy Eval',
            epoch='eval',
            AER=aer,
            ratio=ratio,
        )
        
        # Eval (eval)
        aer, ratio = self.eval(eval=True, soft=False)
        self.board.add_scalars(
            prior='Policy Structure Generalization',
            epoch='eval',
            AER=aer,
            ratio=ratio,
        )
        
        # Eval (change start and goal)
        aer, ratio = self.eval(eval=False, soft=True)  
        self.board.add_scalars(
            prior='Policy Path Generalization',
            epoch='eval',
            AER=aer,
            ratio=ratio,
        )
        
        # Eval (eval with occlusion)
        aer, ratio = self.eval(eval=True, soft=False, occlusion=True) 
        self.board.add_scalars(
            prior='Policy Occlusion Structure Generalization',
            epoch='eval',
            AER=aer,
            ratio=ratio,
        )
        
        # Eval (change start and goal with occlusion)
        aer, ratio = self.eval(eval=False, soft=True, occlusion=True) 
        self.board.add_scalars(
            prior='Policy Occlusion Path Generalization',
            epoch='eval',
            AER=aer,
            ratio=ratio,
        )
        

class GAIL:
    def __init__(
        self, 
        dataset, 
        game, 
        maze_path,
        log_name,
        folder='./tmp/gail/', 
        batch_size=32, 
        pretrain=2048
    ) -> None:
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

        self.eval_callback = EvalCallback(
            self.original_game,
            maze_path=maze_path,
            log_name=log_name,
            eval_freq=1,
            determionistic=True,
            render=True,
        )

        if not os.path.exists(folder):
            os.makedirs(folder)

    def run(self, total_timesteps=1e6):
        self.model.train(int(total_timesteps), callback=self.eval_callback)


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
