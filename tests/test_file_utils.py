import unittest

import numpy as np
from unittest.mock import patch, mock_open

from src.labyrinth.file_utils import (
    split, convert, get_local_position, get_nodes, find_edges_start_and_end,
    find_ice_floors, find_key_and_lock, invoke_interpreter, convert_from_file,
    create_file_from_environment, write_into_file
)


PATH = "/".join(__file__.split("/")[:-1])


class TestLabyrinthModule(unittest.TestCase):
    def test_split(self):
        self.assertEqual(list(split([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])

    def test_convert(self):
        self.assertEqual(convert(3, 5, 4), 9)
        self.assertEqual(convert(7, 5, 4), 1)

    def test_get_local_position(self):
        self.assertEqual(get_local_position(7, 5), [1, 2])

    def test_get_nodes(self):
        self.assertEqual(get_nodes((3, 3)), [0, 2, 6, 8])

    def test_find_edges_start_and_end(self):
        module = invoke_interpreter(f"{PATH}/assets/structure_test.labyrinth")
        labyrinth = module.labyrinth

        labyrinth = np.array(labyrinth[::-1])
        labyrinth_shape = labyrinth.shape
        labyrinth_original_shape = (labyrinth_shape[0] + 1) // 2
        labyrinth_original_shape = (labyrinth_original_shape, labyrinth_original_shape)

        vector_labyrinth = labyrinth.reshape((-1))
        nodes = get_nodes(labyrinth_shape)

        edges, start, end = find_edges_start_and_end(
            nodes, vector_labyrinth, labyrinth, labyrinth_original_shape, labyrinth_shape
        )
        self.assertEqual(start, [0, 0])
        self.assertEqual(end, [9, 9])
        self.assertTrue(isinstance(edges, list))
        self.assertEqual(edges, [
            (0, 19), (2, 3), (4, 5), (6, 25), (6, 7), (8, 9), (10, 29), (12, 31),
            (12, 13), (14, 15), (16, 17), (18, 37), (38, 57), (40, 59), (40, 41),
            (42, 43), (46, 65), (46, 47), (50, 51), (52, 53), (54, 73), (56, 75),
            (76, 77), (80, 99), (82, 101), (82, 83), (86, 87), (88, 107), (88, 89),
            (90, 109), (94, 113), (114, 133), (116, 135), (116, 117), (118, 119),
            (122, 141), (122, 123), (124, 143), (124, 125), (128, 129), (130, 131),
            (132, 151), (152, 171), (154, 155), (156, 175), (158, 177), (160, 179),
            (162, 163), (164, 165), (166, 167), (168, 187), (170, 189), (190, 209),
            (190, 191), (194, 213), (196, 215), (198, 199), (200, 201), (202, 221),
            (204, 223), (206, 225), (228, 247), (228, 229), (230, 231), (234, 253),
            (234, 235), (236, 255), (236, 237), (240, 241), (242, 261), (244, 245),
            (246, 265), (266, 267), (268, 269), (270, 289), (274, 293), (276, 295),
            (276, 277), (278, 279), (280, 281), (284, 303), (304, 323), (304, 305),
            (306, 325), (308, 309), (310, 329), (312, 313), (316, 335), (316, 317),
            (318, 319), (320, 321), (322, 341), (344, 345), (346, 347), (348, 349),
            (350, 351), (352, 353), (356, 357), (358, 359)]
        )

    def test_find_ice_floors(self):
        nodes = [0, 2, 4]
        vector_labyrinth = ['I', ' ', ' ', '-', 'I']
        result = find_ice_floors(nodes, vector_labyrinth, (3, 3))
        self.assertEqual(result, [(0, 0), (0, 2)])

    def test_find_key_and_lock(self):
        nodes = [0, 2, 4]
        vector_labyrinth = ['K', ' ', 'D', '-', ' ']
        key, lock = find_key_and_lock(nodes, vector_labyrinth, (3, 3))
        self.assertEqual(key, [0, 0])
        self.assertEqual(lock, [0, 1])

    @patch("builtins.open", new_callable=mock_open, read_data="S\nE\n")
    def test_invoke_interpreter(self, mock_file):
        interpreter = invoke_interpreter("dummy_path")
        self.assertTrue(interpreter)

    def test_convert_from_file(self):
        save_string, variables = convert_from_file(f"{PATH}/assets/structure_test.labyrinth")
        self.assertTrue(isinstance(save_string, str))
        self.assertTrue(isinstance(variables, dict))

    @patch("builtins.open", new_callable=mock_open)
    def test_write_into_file(self, mock_file):
        labyrinth = [['S', '|'], ['-', 'E']]
        write_into_file(labyrinth, "dummy_path")
        mock_file.assert_called_once_with("dummy_path", "w", encoding="utf-8")
