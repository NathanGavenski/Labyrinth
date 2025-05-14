class Interpreter:

    comment_context = False

    available_keys = ["key_and_lock", "icy_floor", "occlusion"]
    variables = {key: False for key in available_keys}
    stack = []

    labyrinth = []
    labyrinth_context = False

    def reset(self) -> None:
        self.labyrinth = []
        self.stack = []

        self.labyrinth_context = False
        self.comment_context = False

        self.variables = {key: False for key in self.available_keys}

    def prepare_tokens(self, expression: str) -> str:
        """Break some characters into tokens.

        Args:
            expression (str): expression to parse.

        Returns:
            parsed_expression (str): expression with tokens replaced.
        """
        expression = expression.replace("\n", " <NEWLINE>")
        expression = expression.replace("\"\"\"", "<COMMENT>")
        expression = expression.replace(":", " <VARIABLE> ")
        expression = expression.replace("labyrinth", " <STRUCTURE> ")
        expression = expression.replace("end", " <END> ")
        return expression

    def eval(self, expression: str) -> None:
        """Evaluate a line of labyrinth-language.

        Args:
            expression (str): expression line.
        """
        expression = self.prepare_tokens(expression)
        expression = expression.split(" ")

        if len(expression) == 0:
            return

        self.eval_tokens(expression)

    def handle_comments(self, tokens: list[str], index: int) -> int:
        """Handle comment sections.

        Args:
            tokens (list[str]): list of tokens.
            index (int): current index.

        Returns:
            index (int): updated index.
        """
        if "<COMMENT>" in tokens[index]:
            self.comment_context = not self.comment_context
        index += 1

        while "<COMMENT>" not in tokens[index] and tokens[index] != "<NEWLINE>":
            index += 1

        if "<COMMENT>" in tokens[index]:
            self.comment_context = not self.comment_context
        index += 1
        return index

    def handle_labyrinth_structure(self, tokens: list[str], index: int) -> int:
        """Handle labyrinth sections.

        Args:
            tokens (list[str]): list of tokens.
            index (int): current index.

        Returns:
            index (int): updated index.
        """
        if tokens[index] == "":
            count, index = 1, index + 1
            while tokens[index] == "":
                count += 1
                index += 1
            self.stack += ["" for _ in range(count // 2)]
            return index
        elif tokens[index] == "<NEWLINE>":
            if len(self.stack) > 0:
                self.labyrinth.append(self.stack[1:-1])
                self.stack = []
            return 999
        else:
            if "---" not in tokens[index]:
                self.stack.append(tokens[index])
            return index + 1

    def eval_tokens(self, tokens: list[str]) -> None:
        """Evaluate sequence of tokens in labyrinth-language.

        Args:
            tokens (list[str]): tokens to evaluate.
        """
        index = 0
        while index < len(tokens):
            if "<COMMENT>" in tokens[index] or self.comment_context:
                index = self.handle_comments(tokens, index)
            elif tokens[index] == "<VARIABLE>":
                # Define a variable
                if len(self.stack) == 0:
                    index += 1
                    continue
                key = self.stack.pop()
                if key not in self.available_keys:
                    continue
                self.variables[key] = True if tokens[index + 2] == "True" else False
                index += 3
            elif tokens[index] == "<STRUCTURE>":
                self.stack = []  # Remove empty space
                self.labyrinth_context = True
                break
            elif tokens[index] == "<END>":
                self.labyrinth_context = False
                break
            elif self.labyrinth_context:
                index = self.handle_labyrinth_structure(tokens, index)
            elif tokens[index] == "<NEWLINE>" or len(tokens) == 2 and tokens[index] == "":
                break
            elif not self.comment_context:
                # Should be added to the stack
                self.stack.append(tokens[index])
                index += 1

    def __str__(self):
        """Print interpreter in a more readable way."""
        labyrinth = ""
        for structure in self.labyrinth:
            labyrinth += f"\t{structure}\n"
        output = "Interpreter(\n    "
        output += f"variables: {self.variables},\n    "
        output += f"stack: {self.stack},\n    labyrinth:\n{labyrinth})"
        return output
