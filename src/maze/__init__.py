"""Gym registration for the Maze environment."""
from gym.envs.registration import register

register(
    id="Maze-v0",
    entry_point="src.maze.maze:Maze",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        'shape': (10, 10),
        'occlusion': False,
        'key_and_door': False,
        'icy_floor': False,
        'render_mode': None
    }
)

register(
    id="MazeScripts-v0",
    entry_point="maze.maze:Maze",
    reward_threshold=1,
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        'shape': (10, 10),
        'occlusion': False,
        'key_and_door': False,
        'icy_floor': False,
        'render_mode': None
    }
)
