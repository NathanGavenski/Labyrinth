from collections import defaultdict
import os
from os import listdir
from os.path import isfile, join
import tarfile
import shutil

from imitation_datasets.dataset.huggingface import baseline_to_huggingface
import numpy as np
import pandas as pd
from PIL import Image


def create_npz(df, name):
    dataset = defaultdict(list)
    df = df.copy().reset_index().drop("index", axis=1)
    for start, end in zip(df[df["episode start"] == 1].index, df[df["episode end"] == 1].index):
        for row in range(start, end+1):
            data = df.iloc[row]
            dataset["obs"].append(f"{int(data['state'])}.npy")
            dataset["actions"].append(int(data["action"]))
            dataset["rewards"].append(data["reward"])
            dataset["episode_starts"].append(data["episode start"])
            dataset["maze"].append(maze_info[int(data["maze version"])])
            if row == end:
                dataset["obs"].append(f"{int(data['next_state'])}.npy")
                dataset["actions"].append(int(data["action"]))
                dataset["rewards"].append(data["reward"])
                dataset["episode_starts"].append(data["episode start"])
                dataset["maze"].append(maze_info[int(data["maze version"])])

    np.savez(name, **dataset)
    return dataset


if __name__ == "__main__":
    maze_name = "5"
    folder = 'train'
    mypath = f'./src/environment/mazes/mazes{maze_name}/{folder}/'

    # Collect information about the maze
    mazes = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    maze_info = []
    for _maze in mazes:
        with open(_maze, "r") as f:
            for line in f:
                maze_info.append(line)

    # Create dataset
    x = np.load(f"./tmp/expert/maze{maze_name}/{folder}/dataset.npy")
    dataframe = pd.DataFrame(
        x,
        columns=[
            "maze version",
            "solution number",
            "state",
            "action",
            "next_state",
            "episode reward",
            "reward",
            "episode start",
            "episode end"
        ]
    )

    shortest_dataframe = dataframe.iloc[:0].copy()
    for idx in range(dataframe["maze version"].max().astype(int) + 1):
        amount_solutions = dataframe[dataframe["maze version"] == idx]["solution number"]
        amount_solutions = amount_solutions.max().astype(int)
        solution_indexes = [
            solution_idx
            for solution_idx in range(amount_solutions + 1)
        ]
        path_len = [
            dataframe[
                (dataframe["maze version"] == idx) &
                (dataframe["solution number"] == solution_idx)
            ].shape[0]
            for solution_idx in solution_indexes
        ]
        index = np.argmin(path_len)
        shortest_dataframe = pd.concat([
            shortest_dataframe,
            dataframe[
                (dataframe["maze version"] == idx) &
                (dataframe["solution number"] == index)
            ]],
            ignore_index=True
        )

    all_routes = create_npz(dataframe, "all_routes.npz")
    single_route = create_npz(dataframe[dataframe["solution number"] == 0], "single_route.npz")
    shortest_route = create_npz(shortest_dataframe, "shortest_route.npz")

    baseline_to_huggingface("./all_routes.npz", "./all_routes.jsonl", keys=list(all_routes.keys()))
    baseline_to_huggingface("./single_route.npz", "./single_route.jsonl", keys=list(single_route.keys()))
    baseline_to_huggingface("./shortest_route.npz", "./shortest_route.jsonl", keys=list(shortest_route.keys()))

    with tarfile.open("dataset.tar.gz", "w:gz") as tar:
        tar.add("all_routes.jsonl")
        tar.add("single_route.jsonl")
        tar.add("shortest_route.jsonl")

    os.remove("all_routes.jsonl")
    os.remove("single_route.jsonl")
    os.remove("shortest_route.jsonl")
    os.remove("all_routes.npz")
    os.remove("single_route.npz")
    os.remove("shortest_route.npz")

    # Create images
    if os.path.exists("./tmp/images/"):
        shutil.rmtree("./tmp/images/")

    if not os.path.exists("./tmp/images/"):
        os.makedirs("./tmp/images/")

    for f in all_routes["obs"]:
        path = f"./tmp/expert/maze{maze_name}/{folder}/{f}"
        Image.fromarray(np.load(path)).save(f"./tmp/images/{f.split('.')[0]}.jpg")

    with tarfile.open("images.tar.gz", "w:gz") as tar:
        tar.add("./tmp/images", "images")

    if os.path.exists("./tmp/huggingface/"):
        shutil.rmtree("./tmp/huggingface/")

    if not os.path.exists("./tmp/huggingface/"):
        os.makedirs("./tmp/huggingface/")

    shutil.move("./dataset.tar.gz", "./tmp/huggingface/dataset.tar.gz")
    shutil.move("./images.tar.gz", "./tmp/huggingface/images.tar.gz")
