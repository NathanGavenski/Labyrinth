import gym
from maze import environment
from tensorboard_wrapper.tensorboard import Tensorboard as Board

from algo.il import GAIL, create_gail_dataset

if __name__ == '__main__':

    for idx in range(5):
        path, file = './dataset/dataset5', 'dataset.npy'
        dataset = create_gail_dataset(path, file, times=1)
        env = gym.make('Maze-v0', shape=(5, 5))

        model = GAIL(
            dataset=dataset,
            game=env,
            maze_path='./maze/environment/mazes/mazes5/',
        )
        model.run()

        board = Board(f'GAIL-{idx}', './tmp/board/', delete=True)

        aer, ratio = model.eval(eval=False, soft=True)
        board.add_scalars(
            prior='Policy Eval',
            epoch='eval',
            AER=aer,
            ratio=ratio
        )
        aer, ratio = model.eval(eval=False, soft=True)
        board.add_scalars(
            prior='Policy Soft Generalization',
            epoch='eval',
            AER=aer,
            ratio=ratio
        )
        aer, ratio = model.eval(eval=True, soft=False)
        board.add_scalars(
            prior='Policy Hard Generalization',
            epoch='eval',
            AER=aer,
            ratio=ratio
        )
