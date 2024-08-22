from collections import defaultdict
from functools import partial
import os
from os import listdir
from os.path import join
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from PIL import Image
import tensorflow as tf
from tensorboard.plugins import projector


def get_mazes() -> dict[str, list]:
    structures = {}
    for _type in ["train", "eval", "test"]:
        maze_paths = "./src/environment/mazes/mazes5"
        structures[_type] = [f for f in listdir(f"{maze_paths}/{_type}")]
    return structures


def distance_mazes(x: np.ndarray) -> np.ndarray:
    return None


def distance_features(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    distance = cdist(x, x, metric="euclidean")
    if normalize:
        return (distance - distance.min()) / (distance.max() - distance.min())
    return distance


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

    with open("mazes.txt", "w") as f:
        for _type in ["train", "eval", "test"]:
            f.write(f"{_type}\n")

            solved = np.array([0, 0])
            for mazes in solved_matrix[_type]:
                f.write(f"{np.array_str(mazes, max_line_width=500)} {np.bincount(mazes.astype(int))}\n")
                solved = solved + np.bincount(mazes.astype(int))

            f.write(f"{str(solved)}\n")
            f.write("\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, plot in enumerate(["train", "eval", "test"]):
        ax = axes[idx]
        im = ax.matshow(metrics[plot]["features"], cmap='coolwarm')

        for (i, j), val in np.ndenumerate(metrics[plot]["features"]):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

        ax.set_xticks(range(10))
        ax.set_xticklabels([f'Var{i+1}' for i in range(10)], rotation=45)
        ax.set_yticks(range(10))
        ax.set_yticklabels([f'Var{i+1}' for i in range(10)])
        ax.set_title(plot.capitalize())

    plt.savefig("metrics.png", dpi=300, bbox_inches="tight")

    for _type in ["train", "eval", "test"]:
        config = projector.ProjectorConfig()
        models = defaultdict(partial(np.ndarray, (0, 512)))
        for key, value in model_features[_type].items():
            for model, embedding in enumerate(value):
                models[model] = np.append(
                    models[model],
                    embedding[None],
                    axis=0
                )
        embeddings = np.concatenate(list(models.values()), axis=0)
        embeddings = tf.Variable(embeddings, name=f"features_{_type}")

        if not os.path.exists(f"./logs/{_type}"):
            os.makedirs(f"./logs/{_type}")

        with open(f"./logs/{_type}/metadata_{_type}.tsv", "w") as f:
            # f.write("model\n")
            for model, embedding in models.items():
                for _ in range(embedding.shape[0]):
                    f.write(f"{model}\n")

        embedding = config.embeddings.add()
        embedding.tensor_name = _type
        embedding.metadata_path = f"./logs/{_type}/metadata_{_type}.tsv"
        writer = tf.summary.create_file_writer("./logs")
        projector.visualize_embeddings(f"./logs/{_type}", config)

        saver = tf.train.Checkpoint(embedding=embeddings)
        saver.save(f"./logs/{_type}/{_type}.ckpt")
