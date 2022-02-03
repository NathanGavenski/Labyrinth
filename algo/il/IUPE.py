
from collections import defaultdict
from ctypes import Union
from re import M
import os
from os import listdir
from os.path import isfile, join
from typing import Optional, Tuple

import gym
from gym.spaces import Discrete
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboard_wrapper.tensorboard import Tensorboard as Board

from .utils import Resnet
from .datasets import get_expert_loader, get_random_loader


class EarlyStopController():
    def __init__(self, amount : Optional[int] = 3) -> None:
        self.amount = amount
        self.states = defaultdict(int)

    def check_position(self, s: np.ndarray) -> bool:
        state = tuple(s)
        self.states[state] += 1
        return self.states[state] == self.amount

    def reset(self) -> None:
        self.states = defaultdict(int)

class IDM(nn.Module):
    def __init__(self, action_size: int, input: Optional[Tuple[int, int, int]] = (3, 224, 224)) -> None:
        super().__init__()
        self.encoder = Resnet(normalize=True)
        with torch.no_grad():
            encoder_output = self.encoder(torch.zeros((1, *input)))

        self.fc_layers = nn.Sequential(
            nn.Linear(encoder_output.shape[-1] * 2, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, action_size)
        )

    def forward(self, state: torch.Tensor, nState: torch.Tensor) -> torch.Tensor:
        s = self.encoder(state)
        nS = self.encoder(nState)
        cat = torch.cat((s, nS), dim=1)
        return self.fc_layers(cat)


class Policy(nn.Module):
    def __init__(self, action_size: int, input: Optional[Tuple[int, int, int]] = (3, 224, 224)) -> None:
        super().__init__()
        self.encoder = Resnet(normalize=True)
        with torch.no_grad():
            encoder_output = self.encoder(torch.zeros((1, *input)))

        self.fc_layers = nn.Sequential(
            nn.Linear(encoder_output.shape[-1], 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(self.encoder(state))
       
class IUPE(nn.Module):
    def __init__(
        self,
        environment: gym.Env,
        maze_path : str,
        width : Optional[int] = 10,
        height: Optional[int] = 10,
        random_dataset: str = None,
        expert_dataset: str = None,
        device: Optional[str]= None,
        amount: Optional[int] = 5000,
        batch_size: Optional[int] = 1,
        verbose: Optional[bool] = False,
        debug: Optional[bool] = False,
        early_stop: Optional[bool] = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Model params
        self.device = device if device is not None else 'cpu'
        self.verbose = verbose
        self.maze_path = maze_path
        self.debug = debug
        self.early_stop = early_stop

        # Env params
        self.environment = environment
        self.width = width
        self.height = height
        self.action_space = self.environment.action_space.n
        self.controller = EarlyStopController(20)
        
        # Method params
        self.iupe_dataset = None
        self.iupe_dataset_eval = None
        self.random_path = random_dataset
        self.amount = amount
        self.batch_size = batch_size

        # IDM
        self.random_dataset, self.random_dataset_eval = get_random_loader(
            random_dataset,
            split=.7,
            batch_size=batch_size
        )
        self.idm = IDM(
            self.action_space, 
            self.random_dataset.dataset[0][0].shape
        ).to(self.device)
        self.idm_criterion = nn.CrossEntropyLoss()
        self.idm_optimizer = optim.Adam(self.idm.parameters(), lr=5e-4)
        self.dataset_original_size = len(self.random_dataset.dataset)

        # Policy
        self.expert_dataset = get_expert_loader(expert_dataset, batch_size)
        self.policy = Policy(
            self.action_space,
            self.random_dataset.dataset[0][0].shape
        ).to(self.device)
        self.policy_criterion = nn.CrossEntropyLoss()
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=5e-4)

        # Tensorboard
        if name is None:
            name = str(self.environment).split('<')[-1].replace('>', '')
        self.board = Board(f'IUPE-{name}', './tmp/board/', delete=True)

    def get_env(self) -> gym.Env:
        return self.environment

    def get_min_reward(self) -> float:
        return (-.1 / (self.width * self.height)) * 1000

    def idm_train(self) -> Tuple[float, float]:
        if self.verbose:
            self.pbar.set_description_str(desc=f'Training IDM', refresh=True)

        if not self.idm.training:
            self.idm.train()

        names, datasets = [], []        
        if self.random_dataset is not None:
            datasets.append(self.random_dataset)
            names.append('random')

        if self.iupe_dataset is not None:
            datasets.append(self.iupe_dataset)
            names.append('alpha')

        acc_t = []
        loss_t = []
        for idx, (dataset, name) in enumerate(zip(datasets, names)):  
            for mini_batch in dataset:
                s, nS, a = mini_batch

                s = s.to(self.device)
                nS = nS.to(self.device)
                a = a.to(self.device)

                if idx == 0:
                    self.board.add_grid(
                        prior='IDM',
                        state=s,
                        next_state=nS
                    )

                if isinstance(self.environment.action_space, Discrete):
                    a = a.long()

                self.idm_optimizer.zero_grad()
                pred = self.idm(s, nS)

                loss = self.idm_criterion(pred, a)
                loss.backward()
                self.idm_optimizer.step()
                loss_t.append(loss.item())
                
                acc = (torch.argmax(pred, 1) == a).sum().item() / a.size(0)
                acc_t.append(acc)

                if self.verbose:
                    self.pbar.update()
                    self.pbar.set_postfix_str(f'Loss: {np.mean(loss_t)} Acc: {np.mean(acc_t)}')

                if self.debug:
                    break

        return np.mean(acc_t), np.mean(loss_t)
            
    def idm_eval(self) -> float:
        if self.verbose:
            self.pbar.set_description_str(desc=f'Evaluation IDM', refresh=True)

        if self.idm.training:
            self.idm.eval()

        names, datasets = [], []        
        if self.random_dataset is not None:
            datasets.append(self.random_dataset_eval)
            names.append('random')

        if self.iupe_dataset is not None:
            datasets.append(self.iupe_dataset_eval)
            names.append('alpha')

        acc_t = []
        for dataset, name in zip(datasets, names):  
            for mini_batch in dataset:
                s, nS, a = mini_batch

                s = s.to(self.device)
                nS = nS.to(self.device)
                a = a.to(self.device)

                if isinstance(self.environment.action_space, Discrete):
                    a = a.long()

                self.idm_optimizer.zero_grad()
                pred = self.idm(s, nS)
                
                acc = (torch.argmax(pred, 1) == a).sum().item() / a.size(0)
                acc_t.append(acc)

                if self.verbose:
                    self.pbar.update()
                    self.pbar.set_postfix_str(f'Acc: {np.mean(acc_t)}')

                if self.debug:
                    break
    
        return np.mean(acc_t)

    def policy_train(self) -> Tuple[float, float]:
        if self.verbose:
            self.pbar.set_description_str(desc="Training Policy", refresh=True)

        if not self.policy.training:
            self.policy.train()

        if self.idm.training:
            self.idm.eval()

        loss_t = []
        acc_t = []
        for mini_batch in self.expert_dataset:
            s, nS, _ = mini_batch

            s = s.to(self.device)
            nS = nS.to(self.device)

            a = self.idm(s, nS)
            action = torch.argmax(a, 1)

            self.policy_optimizer.zero_grad()
            pred = self.policy(s)

            loss = self.policy_criterion(pred, action)
            loss.backward()
            self.policy_optimizer.step()
            loss_t.append(loss.item())

            acc = ((torch.argmax(pred, 1) == action).sum().item() / action.shape[0])
            acc_t.append(acc)

            if self.verbose:
                self.pbar.update()
                self.pbar.set_postfix_str(f'Loss: {np.mean(loss_t)} Acc: {np.mean(acc_t)}')

            if self.debug:
                break

        return np.mean(acc_t), np.mean(loss_t)

    def create_alpha(self) -> float:
        if self.verbose:
            self.pbar.set_description_str(desc='Creating alpha', refresh=True)

        if self.policy.training:
            self.policy.eval()

        ratio = self.generate_expert_traj(
            './dataset/alpha/',
            self.maze_path,
            env=self.environment
        )

        iupe_amount = int(self.amount * ratio)
        random_amount = int(self.dataset_original_size * (1 - ratio))
        self.board.add_scalars(
            prior='Alpha',
            random=random_amount,
            iupe=iupe_amount
        )

        if iupe_amount > 0:
            self.iupe_dataset, self.iupe_dataset_eval = get_random_loader(
                './dataset/alpha/',
                split=.7,
                batch_size=self.batch_size,
                amount=iupe_amount
            )
        else:
            self.iupe_dataset, self.iupe_dataset_eval = None, None

        if random_amount > 0:
            self.random_dataset, self.random_dataset_eval = get_random_loader(
                self.random_path,
                split=.7,
                batch_size=self.batch_size,
                amount=random_amount
            )
        else:
            self.random_dataset, self.random_dataset_eval = None, None
        
        return ratio

    def forward(self, s : np.array, weight: bool = True) -> Tuple[int, Union[float, None]]:
        s = transforms.ToTensor()(s.copy())[None].to(self.device)
        pred = self.policy(s)

        if weight:
            classes = np.arange(self.action_space)
            prob = torch.nn.functional.softmax(pred, dim=1).cpu().detach().numpy()
            a = np.random.choice(classes, p=prob[0])
        else:
            a = torch.argmax(pred, 1).cpu().detach().numpy().squeeze(0)
        
        return a, None

    def evaluate_policy(
        self,
        maze_path: str,
        env: gym.core.Env,
        soft_generalization: bool = False,
        occlusion: bool = False
    ) -> Tuple[float, float]:
        if self.verbose:
            self.pbar.set_description_str(desc='Eval Policy', refresh=True)

        if isinstance(env, str):
            env = gym.make(env, shape=(self.width, self.height))

        mypath = f'{maze_path}{"eval" if not soft_generalization else "train"}/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

        ratio, rewards = 0, []
        for maze in mazes:
            env.load(maze)
            if soft_generalization:
                env.change_start_and_goal(min_distance=(self.width + self.height) // 2)

            if occlusion:
                env.set_occlusion_on()
            else:
                env.set_occlusion_off()

            self.controller.reset()
            done, reward = False, 0

            while not done:
                state = env.render('rgb_array')
                action, _ = self.forward(state, weight=True)
                next_state, r, done, info = env.step(action)
                reward += r

                if self.controller.check_position(info['state']) and self.early_stop:
                    done = True
                    reward = self.get_min_reward()
            
            rewards.append(reward)
            ratio += (info['state'][:2] == env.end).all()

            if self.verbose:
                self.pbar.update()

            if self.debug:
                break
            
            env.close()

        return np.mean(rewards), ratio / len(mazes)

    def generate_expert_traj(
        self,
        path: str,
        maze_path: str,
        env: gym.core.Env
    ) -> float:
        if isinstance(env, str):
            env = gym.make(env, shape=(self.width, self.height))
        
        if not os.path.exists(path):
            os.makedirs(path)

        mypath = f'{maze_path}train/'
        mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

        ratio = 0
        all_rewards = []
        image_idx = 0
        dataset = np.ndarray(shape=[0, 4])
        for maze_idx, maze in enumerate(tqdm(mazes)):
            env.load(maze)
            self.controller.reset()
            done, rewards = False, 0

            while not done:             
                state = env.render('rgb_array')
                np.save(f'{path}{image_idx}', state)
                image_idx += 1             
                
                action, _ = self.forward(state, weight=True)
                next_state, reward, done, info = env.step(action)
                rewards += reward

                entry = [maze_idx, image_idx, action, image_idx + 1]
                dataset = np.append(dataset, np.array(entry)[None], axis=0)

                if done:
                    np.save(f'{path}{image_idx}', env.render('rgb_array'))
                    image_idx += 1

                    all_rewards.append(rewards)
                    ratio += (env.agent == env.end)

                if self.controller.check_position(env.agent) and self.early_stop:
                    ratio += 0
                    done = True
                    all_rewards.append(self.get_min_reward())

            env.close()

            if self.verbose:
                self.pbar.update()
                self.pbar.set_postfix_str(f'Reward: {np.mean(all_rewards)} Solved: {ratio/len(mazes)}%')
   
        np.save(f'{path}/dataset', dataset)
        return ratio/len(mazes)

    def predict(self, s: torch.Tensor) -> torch.Tensor:
        return self.forward(s, weight=False)


    def learn(self, epochs : Optional[int] = 100) -> None:

        for _ in range(epochs):        
            if self.verbose:
                size = 0 if self.random_dataset is None else len(self.random_dataset)
                size += 0 if self.expert_dataset is None else len(self.expert_dataset) 
                size += 0 if self.iupe_dataset is None else len(self.iupe_dataset)         
                self.pbar = tqdm(range(size))

            # ############## IDM ############## #
            idm_acc, idm_loss = self.idm_train()
            self.board.add_scalars(
                prior='IDM Train',
                idm_loss=idm_loss,
                idm_acc=idm_acc
            )

            idm_acc = self.idm_eval()
            self.board.add_scalars(
                prior='IDM Eval',
                idm_acc=idm_acc
            )

            # ############## POLICY ############## #
            policy_acc, policy_loss = self.policy_train()
            self.board.add_scalars(
                prior='Policy',
                policy_loss=policy_loss,
                policy_acc=policy_acc
            )

            # ########## Create New Data ########## #
            ratio = self.create_alpha()
            self.board.add_scalars(
                prior='Alpha',
                ratio=ratio
            )

            # ########## Validate POLICY Structure Generalization ########## #
            aer, ratio = self.evaluate_policy(
                self.maze_path, 
                self.environment
            )
            self.board.add_scalars(
                prior='Structure Generalization',
                aer=aer,
                ratio=ratio
            )
            
            # ########## Validate POLICY Path Generalization ########## #
            aer, ratio = self.evaluate_policy(
                self.maze_path, 
                self.environment,
                soft_generalization=True
            )
            self.board.add_scalars(
                prior='Path Generalization',
                aer=aer,
                ratio=ratio
            )

            # ########## Validate POLICY Occlusion Structure Generalization ########## #
            aer, ratio = self.evaluate_policy(
                self.maze_path,
                self.environment,
                occlusion=True,
            )
            self.board.add_scalars(
                prior='Occlusion Structure Generalization',
                aer=aer,
                ratio=ratio
            )

            # ########## Validate POLICY Occlusion Path Generalization ########## #
            aer, ratio = self.evaluate_policy(
                self.maze_path,
                self.environment,
                soft_generalization=True,
                occlusion=True,
            )
            self.board.add_scalars(
                prior='Occlusion Structure Generalization',
                aer=aer,
                ratio=ratio
            )

            self.board.step()
