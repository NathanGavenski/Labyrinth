from collections import defaultdict
import os
from os import listdir
from os.path import isfile, join
import pickle

import numpy as np


def get_mazes() -> dict[str, list]:
    structures = {}
    for _type in ["train", "eval", "test"]:
        maze_paths = "./src/environment/mazes/mazes5"
        structures[_type] = [f for f in listdir(f"{maze_paths}/{_type}")]
    return structures


if __name__ == "__main__":
    mazes = get_mazes()

    metrics = defaultdict(lambda: defaultdict(list))
    path = "./peter/tmp/bc/Maze/"
    folders = [join(path, f) for f in listdir(path)]
    for folder in folders:
        with open(f"{folder}/stats.pckl", "rb") as handle:
            stats = pickle.load(handle)
        for _type in ["train", "eval", "test"]:
            solutions = stats["solutions"][_type]
            solutions = [solution.split("/")[-1] for solution in solutions]
            metrics[folder][_type] = np.isin(mazes[_type], solutions).astype(int)
        print(metrics)
        break
