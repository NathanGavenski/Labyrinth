from collections import defaultdict
from functools import partial
from os import listdir
from os.path import join
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from PIL import Image


def get_mazes() -> dict[str, list]:
    structures = {}
    for _type in ["train", "eval", "test"]:
        maze_paths = "./src/environment/mazes/mazes5"
        structures[_type] = [f for f in listdir(f"{maze_paths}/{_type}")]
    return structures


def distance_mazes(x: np.ndarray) -> np.ndarray:
    return None


def distance_features(x: np.ndarray) -> np.ndarray:
    return cdist(x, x, metric="euclidean")


def key_to_image(key: tuple[float]) -> Image.Image:
    image = np.array(key).reshape((64, 64, 3))
    image *= 255
    return Image.fromarray(image.astype("uint8"))


if __name__ == "__main__":
    mazes = get_mazes()

    # Mazes
    solved_mazes = defaultdict(lambda: defaultdict(list))
    model_features = defaultdict(lambda: defaultdict(partial(np.ndarray, (0, 512))))

    path = "./peter/tmp/bc/Maze/"
    folders = [join(path, f) for f in listdir(path)]
    folders.sort(key=lambda x: int(x.split("/")[-1]))
    for folder in tqdm(folders):
        with open(f"{folder}/stats.pckl", "rb") as handle:
            stats = pickle.load(handle)
        for _type in ["train", "eval", "test"]:
            solutions = stats["solutions"][_type]
            solutions = [solution.split("/")[-1] for solution in solutions]
            key = int(folder.split("/")[-1])
            solved_mazes[key][_type] = np.isin(mazes[_type], solutions).astype(int)

            features = stats["features"][_type]
            for key, value in features.items():
                model_features[_type][key] = np.append(
                    model_features[_type][key],
                    value["features"][None],
                    axis=0
                )

    solved_matrix = defaultdict(partial(np.ndarray, [0, 100]))
    for _type in ["train", "eval", "test"]:
        for key in range(len(list(solved_mazes.keys()))):
            solved_matrix[_type] = np.append(
                solved_matrix[_type],
                solved_mazes[key][_type][None],
                axis=0
            )

    metrics = defaultdict(lambda: defaultdict(int))
    for _type in tqdm(["train", "eval", "test"]):
        metrics[_type]["solution"] = distance_mazes(solved_matrix[_type])
        metrics[_type]["features"] = np.ndarray((0, 10, 10))
        for key, value in model_features[_type].items():
            metrics[_type]["features"] = np.append(
                metrics[_type]["features"],
                distance_features(value)[None],
                axis=0
            )

        metrics[_type]["features"] = metrics[_type]["features"].mean(0)

    plt.figure(figsize=(8, 6))
    plt.matshow(metrics["train"]["features"], cmap='coolwarm', fignum=1)
    plt.colorbar()

    for (i, j), val in np.ndenumerate(metrics["train"]["features"]):
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    plt.xticks(range(10), [f'Var{i+1}' for i in range(10)], rotation=45)
    plt.yticks(range(10), [f'Var{i+1}' for i in range(10)])

    plt.show()
