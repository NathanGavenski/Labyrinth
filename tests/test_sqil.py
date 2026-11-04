import unittest

import numpy as np

import ablation  # noqa: F401
from methods.sqil import ReplayBuffer, sqil_batch


def filled(capacity, offset, n):
    buffer = ReplayBuffer(capacity)
    for i in range(n):
        buffer.add(offset + i, i % 4, offset + i + 1, i == n - 1)
    return buffer


class TestSQIL(unittest.TestCase):

    def setUp(self):
        self.expert = filled(100, 0, 50)        # states 0..49
        self.agent = filled(100, 1000, 80)      # states 1000..1079
        self.rng = np.random.default_rng(0)

    def test_rewards_and_split(self):
        for batch_size in (256, 255, 8):
            batch = sqil_batch(self.expert, self.agent, batch_size, self.rng)
            half = batch_size // 2
            self.assertEqual(len(batch["states"]), batch_size)
            self.assertTrue((batch["rewards"][:half] == 1).all())
            self.assertTrue((batch["rewards"][half:] == 0).all())
            self.assertTrue((batch["states"][:half] < 50).all())
            self.assertTrue((batch["states"][half:] >= 1000).all())
            self.assertEqual(int(batch["is_expert"].sum()), half)

    def test_batch_keeps_transitions_aligned(self):
        batch = sqil_batch(self.expert, self.agent, 64, self.rng)
        self.assertTrue((batch["next_states"] == batch["states"] + 1).all())
        self.assertTrue((batch["actions"] == (batch["states"] % 1000) % 4).all())

    def test_ring_buffer(self):
        buffer = filled(10, 0, 25)
        self.assertEqual(len(buffer), 10)
        self.assertEqual(sorted(buffer.states.tolist()), list(range(15, 25)))
        restored = ReplayBuffer(1)
        restored.load_state_dict(buffer.state_dict())
        self.assertEqual(restored.states.tolist(), buffer.states.tolist())
        self.assertEqual(restored.position, buffer.position)

    def test_empty_agent_buffer_raises(self):
        with self.assertRaises(ValueError):
            sqil_batch(self.expert, ReplayBuffer(10), 8, self.rng)


if __name__ == "__main__":
    unittest.main()
