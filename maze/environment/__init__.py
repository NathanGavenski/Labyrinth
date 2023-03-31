from gym.envs.registration import register

register(
    id="Maze-v0",
    entry_point="maze.environment.maze:Maze",
    order_enforce=False,
    kwargs={'shape': (10, 10), 'occlusion': False, 'key_and_door': False}
)

register(
    id="MazeScripts-v0",
    entry_point="environment.maze:Maze",
    reward_threshold=1,
    order_enforce=False,
    kwargs={'shape': (10, 10), 'occlusion': False}
)