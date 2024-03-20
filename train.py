from collections import defaultdict
from functools import partial
from os import listdir
from os.path import isfile, join
import types

from benchmark.methods import BC
import gym
from imitation_datasets.dataset import BaselineDataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src import maze


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
                early_stop_count = 0
                while not done:
                    action = self.predict(obs, transforms)
                    next_obs, reward, done, _ = env.step(action)
                    accumulated_reward += reward
                    if (obs == next_obs).all():
                        early_stop_count += 1
                    else:
                        early_stop_count = 0

                    if early_stop_count == 5:
                        step_reward = -.1 / (env.shape[0] * env.shape[1])
                        lower_reward = env.max_episode_steps * step_reward
                        accumulated_reward = lower_reward
                        break
                    obs = next_obs.copy()
            finally:
                env.close()

            success_rate.append(1 if done else 0)
            average_reward.append(accumulated_reward)
        metrics[f"{maze_type} aer"] = np.mean(average_reward)
        metrics[f"{maze_type} aer (std)"] = np.std(average_reward)
        metrics[f"{maze_type} sr"] = np.mean(success_rate)
    metrics["aer"] = metrics["train aer"]
    return metrics


if __name__ == "__main__":
    transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Resize(64),
    ])

    params = {
        "shape": (5, 5),
        "screen_width": 64,
        "screen_height": 64,
        "visual": True,
    }

    dataset = BaselineDataset(
        "NathanGavenski/imagetest",
        source="hf",
        transform=transforms.Resize(64)
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    enjoy = partial(
        enjoy,
        maze_paths="./src/environment/mazes/mazes5",
        maze_settings=params,
        transforms=transform
    )
    method = BC(gym.make("Maze-v0", **params), enjoy_criteria=10, verbose=True)
    method._enjoy = types.MethodType(enjoy, method)
    method.train(100, train_dataset=dataloader)
