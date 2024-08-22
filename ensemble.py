from collections import defaultdict
import os

from benchmark.methods import BC
import gym
import numpy as np
from src import maze
from tqdm import tqdm
from torchvision import transforms


if __name__ == "__main__":
    params = {
        "shape": (5, 5),
        "screen_width": 600,
        "screen_height": 600,
        "visual": True,
        "occlusion": False
    }
    env = gym.make("Maze-v0", **params)
    maze_path = "./src/environment/mazes/mazes5/"

    path = "./peter/tmp/bc/Maze/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
    ])
    folders = [f for f in os.listdir(path)]
    folders.sort(key=lambda x: int(x))

    policies = {}
    for folder in folders:
        weights = f"{path}{folder}/"
        bc = BC(env, config_file="./configs/resnet.yaml")
        bc.load(weights, "best_model")
        policies[folder] = bc

    _maze_path = f"{maze_path}eval"
    structures = [os.path.join(_maze_path, f) for f in os.listdir(_maze_path)]
    success_rate = 0
    solutions = []
    for structure in tqdm(structures):
        env = gym.make("Maze-v0", **params)

        try:
            obs = env.load(structure)
            done = False
            early_stop_count = defaultdict(int)

            while not done:
                actions = []
                for policy in policies.values():
                    actions.append(int(policy.predict(obs, transform)))
                action = max(set(actions), key=actions.count)
                obs, reward, done, _ = env.step(action)
                early_stop_count[tuple(obs.flatten().tolist())] += 1

                if np.max(list(early_stop_count.values())) >= 5:
                    step_reward = -.1 / (env.shape[0] * env.shape[1])
                    lower_reward = env.max_episode_steps * step_reward
                    accumulated_reward = lower_reward
                    break

            success_rate += 1 if done else 0
            solutions.append(1. if done else 0.)
        finally:
            env.close()

    print(success_rate / len(structures))
    print(solutions)
