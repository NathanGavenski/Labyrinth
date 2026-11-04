"""Image-observation Labyrinth environments, the training leakage guard and the expert.

``LabyrinthImageEnv`` runs the unmodified Labyrinth-v0 environment (5x5, 600x600
renders, default 1,000-step cap and reward) and returns observations as uint8
3x64x64 arrays from the shared ImageBank. A rendered frame depends only on the
configuration and the agent tile, so frames are cached by (configuration hash, tile).

``GuardedLabyrinth`` is the only environment training code receives. Every reset
loads one of the allowed (train) configurations, hashes what the environment
actually holds (edges, start, goal) and raises ``LeakageError`` if that hash is not
in the allowed set. Random generation and start/goal changes raise as well.
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from create_expert import state_to_action

from .data import IMAGE_SIZE, ImageBank, Structure, env_key, preprocess

SHAPE = (5, 5)
SCREEN = 600


class LeakageError(RuntimeError):
    """Raised when a training environment would hold a non-train configuration."""


class _Spec:
    """Minimal spec so IL-Datasets methods can read ``spec.name``/``spec.id``."""
    name = "Labyrinth"
    id = "Labyrinth-v0"
    _env_name = "Labyrinth"


class LabyrinthImageEnv:
    """Labyrinth-v0 with cached, preprocessed image observations."""

    spec = _Spec()

    def __init__(self, bank: ImageBank, shape: Tuple[int, int] = SHAPE) -> None:
        self.bank = bank
        self.env = gym.make(
            "Labyrinth-v0", shape=shape, screen_width=SCREEN, screen_height=SCREEN,
            render_mode=None,
        )
        self.action_space = spaces.Discrete(4)
        # HWC like the IL-Datasets visual policies expect; observations are CHW arrays.
        self.observation_space = spaces.Box(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
        self.key: Optional[str] = None
        self.obs_index: int = -1
        self._optimal_cache: Dict[Tuple[str, int], Tuple[Optional[int], Tuple[int, ...]]] = {}

    @property
    def unwrapped(self):
        return self.env.unwrapped

    # -- configuration -------------------------------------------------------
    def load_structure(self, structure: Structure) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load an exact configuration and return the first observation."""
        self.env.unwrapped.load(structure.structure, structure.variables)
        self.key = env_key(self.env)
        return self.observe(), {"key": self.key}

    def refresh_key(self) -> None:
        """Re-hash after an evaluation-only start/goal change."""
        self.key = env_key(self.env)

    # -- observations --------------------------------------------------------
    @property
    def agent_tile(self) -> int:
        env = self.env.unwrapped
        return int(env.get_global_position(env.agent))

    @property
    def goal_tile(self) -> int:
        env = self.env.unwrapped
        return int(env.get_global_position(env.end))

    def _render(self) -> np.ndarray:
        env = self.env.unwrapped
        env.render_mode = "rgb_array"
        try:
            frame = env.render()
        finally:
            env.render_mode = None
        return preprocess(frame)

    def observe(self) -> np.ndarray:
        self.obs_index = self.bank.add(("env", self.key, self.agent_tile), self._render)
        return self.bank.images[self.obs_index]

    # -- gym API -------------------------------------------------------------
    def step(self, action: Any):
        _, reward, terminated, truncated, info = self.env.unwrapped.step(int(action))
        return self.observe(), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    # -- expert --------------------------------------------------------------
    def shortest_path(self, from_tile: Optional[int] = None) -> List[int]:
        """Shortest path from ``from_tile`` (defaults to the start) to the goal.

        ``env.solve(mode="shortest")`` returns the first path its DFS finds, which is
        not the shortest when the labyrinth has loops. The dataset's routes were picked
        as the first minimum-length path of ``env.solve(mode="all")``
        (create_dataset.py), so the expert does the same.
        """
        return self.shortest_paths(from_tile)[0]

    def shortest_paths(self, from_tile: Optional[int] = None) -> List[List[int]]:
        """All minimum-length paths, in the solver's order."""
        env = self.env.unwrapped
        old_start = env.dfs.start
        if from_tile is not None:
            env.dfs.start = from_tile
        try:
            paths = env.solve(mode="all")
        finally:
            env.dfs.start = old_start
        shortest = min(len(path) for path in paths)
        paths = [list(path) for path in paths if len(path) == shortest]
        start = env.dfs.start if from_tile is None else from_tile
        if paths[0][0] != start:
            raise RuntimeError(f"Solver path starts at {paths[0][0]}, expected {start}")
        return paths

    def optimal_actions(self, tile: Optional[int] = None) -> Tuple[Optional[int], Tuple[int, ...]]:
        """(expert action, all optimal actions) from ``tile``; (None, ()) at the goal.

        The expert action is the first step of the first shortest path, the same tie
        break create_dataset.py used for the dataset routes.
        """
        tile = self.agent_tile if tile is None else tile
        cache_key = (self.key, tile)
        if cache_key not in self._optimal_cache:
            if tile == self.goal_tile:
                self._optimal_cache[cache_key] = (None, ())
            else:
                shape = self.env.unwrapped.shape
                actions = [state_to_action(path[0], path[1], shape) for path in self.shortest_paths(tile)]
                self._optimal_cache[cache_key] = (actions[0], tuple(sorted(set(actions))))
        return self._optimal_cache[cache_key]

    def expert_action(self, tile: Optional[int] = None) -> Optional[int]:
        """Expert (shortest-path) action from ``tile`` (defaults to the agent's tile)."""
        return self.optimal_actions(tile)[0]


class GuardedLabyrinth(LabyrinthImageEnv):
    """Training environment restricted to an allowed set of configurations."""

    def __init__(
        self,
        bank: ImageBank,
        structures: Iterable[Structure],
        seed: int = 0,
        shape: Tuple[int, int] = SHAPE,
    ) -> None:
        super().__init__(bank, shape)
        self.structures = list(structures)
        self.allowed = frozenset(structure.key for structure in self.structures)
        if len(self.allowed) != len(self.structures):
            raise ValueError("Duplicate structures in the allowed set.")
        self.rng = np.random.default_rng(seed)
        self._order: List[int] = []
        self.current: Optional[int] = None
        self.resets = 0

    def _verify(self) -> None:
        key = env_key(self.env)
        if key not in self.allowed:
            raise LeakageError(
                "Training environment holds a configuration outside the train set "
                f"(hash {key[:12]})."
            )
        self.key = key

    def load_structure(self, structure: Structure):
        self.env.unwrapped.load(structure.structure, structure.variables)
        self._verify()
        return self.observe(), {"key": self.key}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Load the next train configuration (shuffled cycle without replacement)."""
        if options is not None and options.get("agent", True) is False:
            raise LeakageError("Random labyrinth generation is disabled during training.")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._order = []
        if not self._order:
            self._order = list(self.rng.permutation(len(self.structures)))
        self.current = int(self._order.pop())
        self.resets += 1
        obs, info = self.load_structure(self.structures[self.current])
        info["structure"] = self.current
        return obs, info

    def step(self, action: Any):
        if self.key is None:
            raise LeakageError("step() before reset().")
        return super().step(action)

    def refresh_key(self) -> None:
        raise LeakageError("Start/goal changes are not allowed during training.")

    def change_start(self, *_, **__):
        raise LeakageError("Start changes are not allowed during training.")

    def change_start_and_goal(self, *_, **__):
        raise LeakageError("Start/goal changes are not allowed during training.")

    def agent_random_position(self, *_, **__):
        raise LeakageError("Random agent positions are not allowed during training.")

    def generate(self, *_, **__):
        raise LeakageError("Labyrinth generation is not allowed during training.")

    def state_dict(self) -> Dict[str, Any]:
        return {"rng": self.rng.bit_generator.state, "order": list(self._order)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self._order = list(state["order"])
