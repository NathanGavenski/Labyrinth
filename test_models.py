import argparse
import pickle
from collections import defaultdict
import os
from os import listdir
from os.path import isfile, join
from functools import partial
import types
import gym
from src import maze
import torch
from torchvision import transforms
from benchmark.methods import BC
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from imitation_datasets.dataset import BaselineDataset
from explain import Method


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--features", action="store_true")

    return parser.parse_args()


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
    bc = BC(env, config_file="./configs/resnet.yaml")
    path = "./peter/tmp/bc/Maze/"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(64),
    ])

    if args.discover:
        dataset = BaselineDataset(
            "NathanGavenski/imageeval",
            source="hf",
            hf_split="shortest_route",
            transform=transforms.Resize(64)
        )

        all_acc = {}
        weight_folders = [f for f in listdir(path)]
        for folder in tqdm(weight_folders, desc="Folders"):
            weights = [f for f in listdir(f"{path}{folder}")]
            full_path = f"{path}{folder}/"

            accs = {}
            for weight in weights:
                weight = weight.replace(".ckpt", "")

                bc.load(path=full_path, name=str(weight))
                bc.policy.eval()
                with torch.no_grad():
                    acc = 0
                    for i in range(len(dataset)):
                        s, a, ns = dataset[i]
                        action = torch.argmax(bc.forward(s[None]), dim=1).squeeze()
                        acc += action == a.squeeze()
                accs[weight] = acc / len(dataset)

            for key, value in accs.items():
                if folder in all_acc.keys():
                    if value > all_acc[folder]["value"]:
                        all_acc[folder] = {"key": key, "value": value}
                else:
                    all_acc[folder] = {"key": key, "value": value}
            print(f"Best for folder {folder} is {all_acc[folder]}")
            os.system(f"cp {full_path}{all_acc[folder]['key']}.ckpt {full_path}best_model.ckpt")
        print(all_acc)

    if args.collect:
        maze_paths="./src/environment/mazes/mazes5"
        weight_folders = [f for f in listdir(path)]
        for folder in tqdm(weight_folders, desc="Folders"):
            full_path = f"{path}{folder}/"

            bc.load(path=full_path, name="best_model")
            bc.policy.eval()
            metrics = defaultdict(int)
            solutions = defaultdict(list)
            non_solutions = defaultdict(list)

            with torch.no_grad():
                for maze_type in ["train", "eval", "test"]:
                    _path = f"{maze_paths}/{maze_type}"
                    structures = [join(_path, f) for f in listdir(_path) if isfile(join(_path, f))]

                    average_reward = []
                    success_rate = []

                    structures = tqdm(structures, desc=f"eval with {maze_type}")
                    for structure in structures:
                        env = gym.make("Maze-v0", **params)
                        done = False

                        try:
                            obs = env.load(structure)
                            accumulated_reward = 0
                            early_stop_count = defaultdict(int)
                            while not done:
                                obs = transform(obs)
                                action = torch.argmax(bc.forward(obs[None]), dim=1).squeeze().item()
                                obs, reward, done, _ = env.step(action)
                                accumulated_reward += reward
                                early_stop_count[tuple(obs.flatten().tolist())] += 1

                                if np.max(list(early_stop_count.values())) >= 5:
                                    step_reward = -.1 / (env.shape[0] * env.shape[1])
                                    lower_reward = env.max_episode_steps * step_reward
                                    accumulated_reward = lower_reward
                                    break

                            if done:
                                solutions[maze_type].append(structure)
                            else:
                                non_solutions[maze_type].append(structure)
                        finally:
                            env.close()

                        success_rate.append(1 if done else 0)
                        average_reward.append(accumulated_reward)
                    metrics[f"{maze_type} aer"] = np.mean(average_reward)
                    metrics[f"{maze_type} aer (std)"] = np.std(average_reward)
                    metrics[f"{maze_type} sr"] = np.mean(success_rate)

            with open(f"{full_path}stats.pckl", "wb") as handle:
                stats = {
                    "metrics": metrics,
                    "solutions": solutions,
                    "non solutions": non_solutions
                }
                pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.features:
        weight_folders = [f for f in listdir(path)]
        for folder in tqdm(weight_folders, desc="Folders"):
            full_path = f"{path}{folder}/"

            with open(f"{full_path}stats.pckl", "rb") as handle:
                stats = pickle.load(handle)

            features = {}
            for _type in tqdm(["train", "eval", "test"], desc="type"):
                bc.load(path=full_path, name="best_model")
                bc.policy.eval()
                method = Method(bc, env, transform)
                method.get_retrieval_info()
                features[_type] = method.retrieval_data

                stats["features"] = features

                with open(f"{full_path}stats.pckl", "wb") as handle:
                    pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
