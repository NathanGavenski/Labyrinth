import argparse
from collections import defaultdict
from functools import partial
from os import listdir
from os.path import isfile, join
import types

from benchmark.methods import BC
import gym
from imitation_datasets.dataset import BaselineDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src import maze


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", type=str, required=True)

    return parser.parse_args()


def enjoy(self, maze_paths, maze_settings, transforms):
    metrics = defaultdict(int)

    for maze_type in ["train", "eval"]:
        path = f"{maze_paths}/{maze_type}"
        structures = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

        average_reward = []
        success_rate = []

        if self.verbose:
            structures = tqdm(structures, desc=f"eval with {maze_type}")
        for structure in structures:
            env = gym.make("Maze-v0", **maze_settings)
            done = False

            try:
                obs = env.load(structure)
                accumulated_reward = 0
                early_stop_count = defaultdict(int)
                while not done:
                    action = self.predict(obs, transforms)
                    obs, reward, done, _ = env.step(action)
                    accumulated_reward += reward
                    early_stop_count[tuple(obs.flatten().tolist())] += 1

                    if np.max(list(early_stop_count.values())) >= 5:
                        step_reward = -.1 / (env.shape[0] * env.shape[1])
                        lower_reward = env.max_episode_steps * step_reward
                        accumulated_reward = lower_reward
                        break
            finally:
                env.close()

            success_rate.append(1 if done else 0)
            average_reward.append(accumulated_reward)
        metrics[f"{maze_type} aer"] = np.mean(average_reward)
        metrics[f"{maze_type} aer (std)"] = np.std(average_reward)
        metrics[f"{maze_type} sr"] = np.mean(success_rate)
    metrics["aer"] = metrics["train sr"]
    return metrics


if __name__ == "__main__":
    args = get_args()

    transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Resize(64),
    ])

    params = {
        "shape": (5, 5),
        "screen_width": 600,
        "screen_height": 600,
        "visual": True,
    }

    dataset = BaselineDataset(
        "NathanGavenski/imagetest",
        source="hf",
        hf_split="shortest_route",
        transform=transforms.Resize(64)
    )
    dataset.states = dataset.states.repeat(10).reshape(-1, 1)
    dataset.next_states = dataset.next_states.repeat(10).reshape(-1, 1)
    dataset.actions = torch.from_numpy(dataset.actions.numpy().repeat(10)).view((-1, 1))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    enjoy = partial(
        enjoy,
        maze_paths="./src/environment/mazes/mazes5",
        maze_settings=params,
        transforms=transform
    )

    env = gym.make("Maze-v0", **params)
    method = BC(env, enjoy_criteria=10, verbose=True, config_file=args.file)
    print(method.hyperparameters)

    method._enjoy = types.MethodType(enjoy, method)
    method.train(101, train_dataset=dataloader, always_save=True)
