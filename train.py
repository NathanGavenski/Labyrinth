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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_models", type=int, default=10)
    parser.add_argument("--n_early_stop", type=int, default=5)

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

    # Metric we use to save best model if not always_save
    metrics["aer"] = metrics["eval sr"]

    if self.best_model < metrics["aer"]:
        self.best_model = metrics["aer"]
        self.early_stop_count = 0
    else:
        self.early_stop_count += 1

    return metrics


def early_stop(self, metric, n_early_stop) -> bool:
    if self.early_stop_count == n_early_stop:
        print("Early stop triggered")
        return True
    return False


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

    train_dataset = BaselineDataset(
        "NathanGavenski/imagetrain",
        source="hf",
        hf_split="shortest_route",
        transform=transforms.Resize(64)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    eval_dataset = BaselineDataset(
        "NathanGavenski/imageeval",
        source="hf",
        hf_split="shortest_route",
        transform=transforms.Resize(64)
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

    enjoy = partial(
        enjoy,
        maze_paths="./src/environment/mazes/mazes5",
        maze_settings=params,
        transforms=transform
    )
    early_stop = partial(
        early_stop,
        n_early_stop=args.n_early_stop
    )

    env = gym.make("Maze-v0", **params)

    for model in range(args.n_models):
        method = BC(env, enjoy_criteria=10, verbose=True, config_file=args.file)

        # Things for overwriting default IL-Datasets
        method.best_model = -np.inf
        method.early_stop_count = 0
        method.save_path = f"./tmp/bc/{method.environment_name}/{model}"

        method._enjoy = types.MethodType(enjoy, method)
        method.early_stop = types.MethodType(early_stop, method)

        # Start IL-Datasets training
        method.train(
            100 * 10 + 1,
            train_dataset=train_dataloader,
            eval_dataset=eval_dataloader,
            always_save=True
        )
        print(f"{model + 1} out of {args.n_models} finished training")
