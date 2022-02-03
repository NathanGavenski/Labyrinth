import gym
from maze import environment
from tensorboard_wrapper.tensorboard import Tensorboard as Board

from algo.il import GAIL, create_gail_dataset

if __name__ == '__main__':

    for idx in range(5):
        path, file = './dataset/dataset5', 'dataset.npy'
        dataset = create_gail_dataset(path, file, times=100)
        env = gym.make('Maze-v0', shape=(5, 5))

        model = GAIL(
            dataset=dataset,
            game=env,
            maze_path='./maze/environment/mazes/mazes5/',
        )
        model.run()
