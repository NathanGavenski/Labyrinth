import unittest
from unittest.mock import patch
import numpy as np
import argparse
from src.create_random import get_args, create


class TestCreateRandomDataset(unittest.TestCase):

    @patch("argparse.ArgumentParser.parse_args", return_value=argparse.Namespace(
        verbose=False,
        save_path="/fake/path",
        path="/fake/data",
        amount=100,
        unbiased=False,
        random_start=False,
        times=10,
        width=10,
        height=10
    ))
    def test_get_args(self, mock_args):
        args = get_args()
        self.assertEqual(args.save_path, "/fake/path")
        self.assertEqual(args.path, "/fake/data")
        self.assertEqual(args.amount, 100)
        self.assertFalse(args.unbiased)

    def test_create(self):
        args = argparse.Namespace(
            verbose=True,
            save_path="./tests/tmp",
            path="./tests/assets",
            amount=10,
            unbiased=False,
            random_start=False,
            times=1,
            width=10,
            height=10
        )

        dataset = create(args)
        self.assertIsInstance(dataset, np.ndarray)
        self.assertEqual(dataset.shape[1], 4)
