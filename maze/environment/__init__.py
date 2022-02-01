from gym.envs.registration import register

register(
    id="Maze-v0",
    entry_point="maze.environment.maze:Maze",
    reward_threshold=1,
    kwargs={'shape': (10, 10)}
)

register(
    id="MazeScripts-v0",
    entry_point="environment.maze:Maze",
    reward_threshold=1,
    kwargs={'shape': (10, 10)}
)