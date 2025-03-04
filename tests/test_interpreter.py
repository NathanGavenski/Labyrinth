from unittest import TestCase

from src.maze.interp import Interpreter


PATH = "/".join(__file__.split("/")[:-1])


class TestInterpreter(TestCase):
    """Test cases for testing Interpreter class."""

    def setUp(self) -> None:
        self.interpreter = Interpreter()
        self.interpreter.reset()

    def test_reset(self) -> None:
        self.interpreter.variables["key_and_lock"] = True
        self.interpreter.stack.append("test")
        self.interpreter.maze.append(["row"])

        self.interpreter.reset()

        assert all(value is False for value in self.interpreter.variables.values())
        assert self.interpreter.stack == []
        assert self.interpreter.maze == []
        assert not self.interpreter.maze_context
        assert not self.interpreter.comment_context

    def test_prepare_tokens(self) -> None:
        expression = '"""comment"""\nkey_and_lock: True\nmaze:\nend'
        expected = '<COMMENT>comment<COMMENT> <NEWLINE>'
        expected += 'key_and_lock <VARIABLE>  True <NEWLINE> '
        expected += '<STRUCTURE>  <VARIABLE>  <NEWLINE> <END> '
        assert self.interpreter.prepare_tokens(expression) == expected

    def test_eval_empty(self) -> None:
        self.interpreter.eval("")
        assert self.interpreter.stack == ['']
        assert self.interpreter.maze == []

    def test_eval_variable_assignment(self) -> None:
        self.interpreter.eval("key_and_lock : True")
        assert self.interpreter.variables["key_and_lock"] is True

        self.interpreter.eval("icy_floor : False")
        assert self.interpreter.variables["icy_floor"] is False

    def test_eval_invalid_variable(self) -> None:
        self.interpreter.eval("non_existent : True")
        assert "non_existent" not in self.interpreter.variables

    def test_handle_comments(self) -> None:
        tokens = ["<COMMENT>", "This", "is", "a", "comment", "<COMMENT>"]
        index = self.interpreter.handle_comments(tokens, 0)
        assert index == len(tokens)
        assert not self.interpreter.comment_context

    def test_handle_comments_new_line(self) -> None:
        tokens = ["<COMMENT>", "This", "is", "a", "comment", "<NEWLINE>"]
        index = self.interpreter.handle_comments(tokens, 0)
        assert index == len(tokens)
        assert self.interpreter.comment_context

    def test_handle_maze_structure(self) -> None:
        tokens = ["", "", "|", "-", "+"]
        index = self.interpreter.handle_maze_structure(tokens, 0)
        assert self.interpreter.stack == [""]
        assert index == 2

    def test_eval_tokens_with_comment(self) -> None:
        tokens = [
            "<COMMENT>", "<NEWLINE>", "ignored", "<NEWLINE>", "<COMMENT>", "<NEWLINE>",
            "key_and_lock", "<VARIABLE>", " ", "True"
        ]
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.variables["key_and_lock"] is True

    def test_eval_tokens_with_maze_structure(self) -> None:
        from src.maze.file_utils import convert_from_file
        convert_from_file(
            f"{PATH}/assets/occlusion_vector_test.maze"
        )
        tokens = ["<STRUCTURE>"]
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.maze_context is True
        assert self.interpreter.maze == []
        assert self.interpreter.stack == []

        tokens = ['|', '', '', '', '', '', '', '', '', 'E', '|', '<NEWLINE>']
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.maze_context is True
        assert self.interpreter.maze == [['', '', '', '', 'E']]
        assert self.interpreter.stack == []

        tokens = ['|', '', '', '+', '', '', '+', '', '', '|', '<NEWLINE>']
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.maze_context is True
        assert self.interpreter.maze == [
            ['', '', '', '', 'E'],
            ['', '+', '', '+', '']
        ]
        assert self.interpreter.stack == []

        tokens = ['|', '', '', '+', '', '', '+', '', '', '|', '<NEWLINE>']
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.maze_context is True
        assert self.interpreter.maze == [
            ['', '', '', '', 'E'],
            ['', '+', '', '+', ''],
            ['', '+', '', '+', ''],
        ]
        assert self.interpreter.stack == []

        tokens = ['|', '-', '+', '-', '+', '', '', '|', '<NEWLINE>']
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.maze_context is True
        assert self.interpreter.maze == [
            ['', '', '', '', 'E'],
            ['', '+', '', '+', ''],
            ['', '+', '', '+', ''],
            ['-', '+', '-', '+', ''],
        ]
        assert self.interpreter.stack == []

        ['|', 'S', '', '', '', '', '', '', '', '', '|', '<NEWLINE>']
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.maze_context is True
        assert self.interpreter.maze == [
            ['', '', '', '', 'E'],
            ['', '+', '', '+', ''],
            ['', '+', '', '+', ''],
            ['-', '+', '-', '+', ''],
            ['-', '+', '-', '+', ''],
        ]
        assert self.interpreter.stack == []

        ['-------------', '<NEWLINE>']
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.maze_context is True
        assert self.interpreter.maze == [
            ['', '', '', '', 'E'],
            ['', '+', '', '+', ''],
            ['', '+', '', '+', ''],
            ['-', '+', '-', '+', ''],
            ['-', '+', '-', '+', ''],
            ['-', '+', '-', '+', '']
        ]
        assert self.interpreter.stack == []

        ['', '<END>', '']
        self.interpreter.eval_tokens(tokens)
        assert self.interpreter.maze_context is True
        print(self.interpreter.maze)
        assert self.interpreter.maze == [
            ['', '', '', '', 'E'],
            ['', '+', '', '+', ''],
            ['', '+', '', '+', ''],
            ['-', '+', '-', '+', ''],
            ['-', '+', '-', '+', ''],
            ['-', '+', '-', '+', ''],
            ['-', '+', '-', '+', '']
        ]
        assert self.interpreter.stack == []

    def test_str_representation(self) -> None:
        self.interpreter.variables["key_and_lock"] = True
        self.interpreter.stack.append("test")
        self.interpreter.maze.append(["row"])
        expected_output = (
            "Interpreter(\n    "
            "variables: {'key_and_lock': True, 'icy_floor': False, 'occlusion': False},\n    "
            "stack: ['test'],\n    maze:\n\t['row']\n)"
        )
        assert str(self.interpreter) == expected_output
