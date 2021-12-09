import gym

from maze import environment
from algo.il import IUPE

if __name__ == '__main__':
    env = gym.make('Maze-v0', shape=(10, 10))
    algo = IUPE(
        environment=env,
        maze_path='./maze/environment/mazes/mazes10/',
        width=10,
        height=10,
        random_dataset='./dataset/random_dataset10/',
        expert_dataset='./dataset/dataset10/',
        device='cpu',
        batch_size=8,
        verbose=True
    )
    algo.learn(1)