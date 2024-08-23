import argparse
from collections import defaultdict
import os

from benchmark.methods import BC
import gym
import numpy as np
from src import maze
from tqdm import tqdm
import torch
from torchvision import transforms
from explain import Method


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--strategy", default="democratic", required=True)
    parser.add_argument("--type", default="train", required=True)
    return parser.parse_args()


class Ensemble:
    def __init__(
        self,
        policies: list[BC],
        env: gym.Env,
        transform: transforms.Compose,
        strategy: str = "democratic",
    ) -> None:
        self.policies: list[BC] = policies
        self.strategy: str = strategy
        self.features: dict[int, Method] = {}
        if strategy not in ["democratic", "confidence"]:
            for i, policy in enumerate(policies):
                self.features[i] = Method(policy, env, transform, "train")

    def predict_action(
        self,
        obs: np.ndarray,
        transform: transforms.Compose,
    ) -> int:
        if self.strategy == "democratic":
            actions = []
            for policy in self.policies:
                actions.append(int(policy.predict(obs, transform)))
            return max(set(actions), key=actions.count)

        if self.strategy == "confidence":
            actions = np.ndarray((0, 4))
            obs = transform(obs)[None]
            for policy in self.policies:
                policy.policy.eval()
                action = policy.forward(obs).detach()
                action = torch.softmax(action, dim=1).numpy()
                actions = np.append(actions, action, axis=0)
            actions = actions.sum(0)
            return np.argmax(actions)

        if self.strategy == "knn":
            actions = []
            for features, policy in zip(self.features.values(), self.policies):
                weight = 1 / features.retrieve_distance(obs)
                action = int(policy.predict(obs, transform))
                actions.append([weight, action])

            actions = np.array(actions)
            actions[:, 0] = actions[:, 0] / actions[:, 0].sum()

            _actions = np.zeros((0, 4))
            for data in actions:
                _action = np.zeros((4,))
                _action[int(data[1])] = data[0]
                _actions = np.append(_actions, _action[None], axis=0)
            actions = _actions.sum(axis=0)
            return np.argmax(actions)

        if self.strategy == "kmeans":
            raise NotImplementedError()

        return None


if __name__ == "__main__":
    args = get_args()

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
    policies = Ensemble(list(policies.values()), env, transform, strategy=args.strategy)

    _maze_path = f"{maze_path}{args.type}"
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
                action = policies.predict_action(obs, transform)
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
