"""Gym registration for the Labyrinth environment."""
from gymnasium.envs.registration import register


register(
    id="Labyrinth-v0",
    entry_point="labyrinth.labyrinth:Labyrinth",
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
