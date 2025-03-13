from os import listdir
from os.path import join

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import maze
from src.maze.file_utils import convert_from_file
from src.create_expert import state_to_action


if __name__ == "__main__":

    for size in [5, 10, 25, 50, 75, 100]:
        tile_distribution = {}
        action_distribution = {}
        path = f"./src/environment/mazes/mazes{size}/"
        for folder in ["train", "eval", "test"]:
            tiles = {x: 0 for x in range(size * size)}
            actions = {x: 0 for x in range(4)}
            maze_files = [
                join(join(path, folder), f)
                for f in listdir(join(path, folder))
            ]

            env = gym.make("Maze-v0", shape=(size, size)) # , render_mode="human", screen_width=1280, screen_height=1280)

            for maze_file in tqdm(maze_files, desc=f"Maze-v0 {size}x{size} - {folder.capitalize()}"):
                env.load(*convert_from_file(maze_file))
                solutions = env.solve(mode="shortest")
                for solution in solutions:
                    obs, _ = env.load(*convert_from_file(maze_file))
                    for idx, tile in enumerate(solution[:-1]):
                        tiles[tile] += 1
                        action = state_to_action(tile, solution[idx + 1], shape=(size, size))
                        actions[action] += 1
                        obs, reward, done, terminated, info = env.step(action)
                    tiles[solution[-1]] += 1

            tile_distribution[folder] = tiles
            action_distribution[folder] = actions

        scale_factor = max(1, size // 10)  # Increase size for larger mazes
        fig, axes = plt.subplots(2, 3, figsize=(15 + scale_factor, 10 + scale_factor))

        # First row: Action distribution
        categories = list(action_distribution.keys())
        actions = list(action_distribution["train"].keys())
        for i, category in enumerate(categories):
            values = [action_distribution[category][a] for a in actions]
            axes[0, i].bar(actions, values, color=['b', 'g', 'r', 'c'])
            axes[0, i].set_title(f"{category.capitalize()} Set")
            axes[0, i].set_xlabel("Actions")
            axes[0, i].set_xticks(actions)
            axes[0, i].set_xticklabels([f"Action {a}" for a in actions])

        # Second row: Tile distribution
        categories = list(tile_distribution.keys())
        train_heatmap = np.zeros((size, size))
        for tile, count in tile_distribution["train"].items():
            row, col = divmod(tile, size)
            train_heatmap[row, col] = count
        train_heatmap = train_heatmap[::-1]

        for i, category in enumerate(categories):
            heatmap = np.zeros((size, size))
            for tile, count in tile_distribution[category].items():
                row, col = divmod(tile, size)
                heatmap[row, col] = count
            heatmap = heatmap[::-1]

            ax = axes[1, i]
            img = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
            ax.set_title(f"{category.capitalize()} Set")
            ax.set_xticks(range(size))
            ax.set_yticks(range(size))
            ax.set_xticklabels(range(1, size + 1))
            ax.set_yticklabels(range(1, size + 1))
            fig.colorbar(img, ax=ax)

        # Save both plots in one image
        plt.tight_layout()
        plt.savefig(f"maze{size}_stats.png", dpi=300)
        print()
