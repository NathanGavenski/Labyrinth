from gym.envs.registration import register

register(
    id="Maze-v0",
    entry_point="environment.maze:Maze",
    max_episode_steps=1000,
    reward_threshold=1,
    kwargs={'shape': (10, 10), 'start': (0, 0), 'end': (9, 9)}
)
