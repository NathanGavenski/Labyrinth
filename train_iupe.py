import gym

from maze import environment
from algo.il import IUPE

if __name__ == '__main__':
    env = gym.make('Maze-v0', shape=(5, 5))
    algo = IUPE(
        environment=env,
        maze_path='./maze/environment/mazes/mazes5/',
        width=5,
        height=5,
        random_dataset='./dataset/random_dataset5/',
        expert_dataset='./dataset/dataset5/',
        device='cuda',
        batch_size=1,
        verbose=True,
        debug=True
    )
    algo.learn(1)
