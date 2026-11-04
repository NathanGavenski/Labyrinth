import os
import unittest
from collections import deque

import ablation  # noqa: F401
from ablation.data import ImageBank, Structure, wall_similarity
from ablation.envs import GuardedLabyrinth, LabyrinthImageEnv, LeakageError


PATH = os.path.join(ablation.ROOT, "src", "environment", "labyrinths", "labyrinths4")


def structures(split):
    folder = os.path.join(PATH, split)
    return [Structure.from_file(os.path.join(folder, f)) for f in sorted(os.listdir(folder))]


def bfs_distance(pathways, start, goal):
    distances, queue = {start: 0}, deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in pathways[node]:
            if neighbour not in distances:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances[goal]


class TestLeakageGuard(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train = structures("train")[:10]
        cls.eval = structures("eval")[:3]

    def setUp(self):
        self.env = GuardedLabyrinth(ImageBank(), self.train, seed=0, shape=(4, 4))

    def test_reset_cycles_train_structures_only(self):
        keys = []
        for _ in range(len(self.train)):
            self.env.reset()
            keys.append(self.env.key)
        self.assertEqual(set(keys), {s.key for s in self.train})

    def test_eval_structure_raises(self):
        for structure in self.eval:
            with self.assertRaises(LeakageError):
                self.env.load_structure(structure)

    def test_changed_start_raises(self):
        self.env.reset()
        self.env.unwrapped.start = (1, 1)
        with self.assertRaises(LeakageError):
            self.env._verify()  # pylint: disable=protected-access

    def test_random_generation_and_position_changes_raise(self):
        with self.assertRaises(LeakageError):
            self.env.reset(options={"agent": False})
        self.env.reset()
        for name in ("change_start", "change_start_and_goal", "agent_random_position", "generate"):
            with self.assertRaises(LeakageError):
                getattr(self.env, name)()

    def test_step_before_reset_raises(self):
        with self.assertRaises(LeakageError):
            self.env.step(0)

    def test_splits_disjoint(self):
        self.assertFalse({s.key for s in self.train} & {s.key for s in self.eval})

    def test_wall_similarity(self):
        self.assertEqual(wall_similarity(self.train[0], self.train[0]), 1.0)
        self.assertLess(wall_similarity(self.train[0], self.eval[0]), 1.0)


class TestExpert(unittest.TestCase):

    def test_expert_paths_are_shortest(self):
        env = LabyrinthImageEnv(ImageBank(), shape=(4, 4))
        for structure in structures("train")[:20]:
            env.load_structure(structure)
            pathways = env.unwrapped.pathways
            for tile in range(16):
                if tile == env.goal_tile:
                    continue
                path = env.shortest_path(tile)
                self.assertEqual(path[0], tile)
                self.assertEqual(path[-1], env.goal_tile)
                self.assertEqual(len(path) - 1, bfs_distance(pathways, tile, env.goal_tile))


if __name__ == "__main__":
    unittest.main()
