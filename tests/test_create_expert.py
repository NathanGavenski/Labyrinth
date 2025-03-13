import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import argparse

from src.create_expert import state_to_action, get_args, create


class TestCreateTeacherDataset(unittest.TestCase):
    
    def test_state_to_action(self):
        shape = (5, 5)
        self.assertEqual(state_to_action(0, 1, shape), 1)
        self.assertEqual(state_to_action(1, 0, shape), 3)
        self.assertEqual(state_to_action(0, 5, shape), 0)
        self.assertEqual(state_to_action(5, 0, shape), 2)

    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(
        verbose=True, save_path='/tmp/', path='/tmp/', unbiased=False,
        times=10, folder='train', width=10, height=10, shortest=False
    ))
    def test_get_args(self, mock_parse_args):
        args = get_args()
        self.assertTrue(args.verbose)
        self.assertEqual(args.save_path, '/tmp/')
        self.assertEqual(args.path, '/tmp/')
        self.assertFalse(args.unbiased)
        self.assertEqual(args.times, 10)
        self.assertEqual(args.folder, 'train')
        self.assertEqual(args.width, 10)
        self.assertEqual(args.height, 10)
        self.assertFalse(args.shortest)

    @patch('os.listdir', return_value=['maze1.txt', 'maze2.txt'])
    @patch('os.path.isfile', return_value=True)
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    @patch('shutil.rmtree')
    @patch('numpy.save')
    @patch('maze.file_utils.convert_from_file', return_value=([], []))
    @patch('gymnasium.make')
    def test_create(
        self, mock_gym_make, mock_convert, mock_save, mock_rmtree, mock_makedirs,
        mock_exists, mock_isfile, mock_listdir
    ):
        mock_env = MagicMock()
        mock_env.solve.return_value = [[0, 1, 2, 3]]
        mock_env.render.return_value = np.zeros((10, 10, 3))
        mock_env.step.return_value = (None, 1, False, False, None)
        mock_gym_make.return_value = mock_env

        args = argparse.Namespace(
            verbose=True, save_path='./tests/tmp/', path='./tests/assets', unbiased=False,
            times=10, folder='train', width=10, height=10, shortest=False
        )
        dataset = create(args, folder='train')
        self.assertEqual(dataset.shape[1], 10)
        self.assertGreater(dataset.shape[0], 0)
