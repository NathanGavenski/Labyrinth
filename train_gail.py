import gym
from maze import environment
from tensorboard_wrapper.tensorboard import Tensorboard as Board

from algo.il import GAIL, create_gail_dataset

if __name__ == '__main__':

    print('Creating dataset')
    path, file = './dataset/dataset5', 'dataset.npy'
    dataset = create_gail_dataset(path, file, times=100)

    for idx in range(5):
        print(f'Running {idx}')
        env = gym.make('Maze-v0', shape=(5, 5))
        model = GAIL(
            dataset=dataset,
            game=env,
            maze_path='./maze/environment/mazes/mazes5/',
            log_name=idx
        )
        model.run()
