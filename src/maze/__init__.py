"""Gym registration for the Maze environment."""
from gymnasium.envs.registration import register

register(
    id="Maze-v0",
    entry_point="maze.maze:Maze",
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
