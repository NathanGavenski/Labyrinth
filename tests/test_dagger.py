import os
import tempfile
import unittest

import torch

import ablation  # noqa: F401
from ablation.data import ImageBank, IndexedTransitions, Structure
from ablation.envs import GuardedLabyrinth
from create_expert import state_to_action
from methods.base import TrainingContext
from methods.dagger import DAgger, beta_schedule, round_starts

# README example: the only path is 0 -> 3 -> 4 -> 7 -> 8 (UP, RIGHT, UP, RIGHT).
KNOWN = '''"""
This file was created automatically.
For more instructions read the README.md
"""
key_and_lock: False
icy_floor: False
occlusion: False

labyrinth:
-------------
|   |     E |
| - +   + - |
|       |   |
|   + - +   |
| S |   |   |
-------------
end'''

CONFIG = """Default:
  lr: 5e-4
  policy: 'CnnPolicy'
  hidden_dim: [512, 512]
  rounds: 2
  beta_decay: 0.5
  rollout_horizon: 50
"""


class TestDAgger(unittest.TestCase):

    def setUp(self):
        self.structure = Structure.from_info(KNOWN)
        self.bank = ImageBank()
        self.env = GuardedLabyrinth(self.bank, [self.structure], seed=0, shape=(3, 3))
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as _file:
            _file.write(CONFIG)
            self.config = _file.name
        self.method = DAgger(self.env, config_file=self.config)
        self.env.reset()
        dataset = IndexedTransitions(
            self.bank, torch.tensor([self.env.obs_index]), torch.tensor([0]),
            torch.tensor([self.env.obs_index]), torch.tensor([False]),
        )
        self.method.attach(TrainingContext(
            env=self.env, bank=self.bank,
            expert_loader=torch.utils.data.DataLoader(dataset, batch_size=4),
            idm_loader=None, n_epochs=4, batch_size=4, device="cpu", rollout_episodes=1,
        ))

    def tearDown(self):
        os.remove(self.config)

    def tile_of(self, bank_index):
        keys = {idx: key for key, idx in self.bank.index.items()}
        return keys[bank_index][2]

    def test_solver_on_known_labyrinth(self):
        self.env.reset()
        self.assertEqual(self.env.shortest_path(), [0, 3, 4, 7, 8])
        self.assertEqual(
            [self.env.expert_action(tile) for tile in (0, 3, 4, 7)], [0, 1, 0, 1]
        )

    def test_expert_rollout_labels_match_solver(self):
        states, labels, info = self.method.collect(beta=1.0)
        self.assertEqual([self.tile_of(s) for s in states], [0, 3, 4, 7])
        self.assertEqual(labels, [0, 1, 0, 1])
        self.assertEqual(info["mixture_success_rate"], 1.0)

    def test_policy_rollout_labels_match_solver_from_visited_tiles(self):
        states, labels, _ = self.method.collect(beta=0.0)
        self.assertGreater(len(states), 0)
        for state, label in zip(states, labels):
            tile = self.tile_of(state)
            path = self.env.shortest_path(tile)
            self.assertEqual(label, state_to_action(path[0], path[1], (3, 3)))

    def test_rounds_and_aggregation(self):
        self.assertEqual(round_starts(1000, 10), list(range(0, 1000, 100)))
        self.assertEqual(round_starts(1, 10), [0])
        self.assertEqual(beta_schedule(3, 0.5), [1.0, 0.5, 0.25])
        for epoch in range(4):
            self.method.run_epoch(epoch)
        self.assertEqual(len(self.method.collected), 1)
        self.assertEqual(self.method.collected[0]["epoch"], 2)
        self.assertEqual(len(self.method.aggregated), 1 + self.method.collected[0]["states"])


if __name__ == "__main__":
    unittest.main()
