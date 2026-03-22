"""Parser for the Kata language — tokens → AST."""

from __future__ import annotations
from .lexer import Token, TokenKind
from .ast import (
    Program,
    DirectiveNode,
    ModelDirective,
    RoleDirective,
    ContextDirective,
    TaskDirective,
    OutputDirective,
    ConstraintDirective,
    FnDirective,
    RetryDirective,
    ImportDirective,
    CallDirective,
    Span,
    Position,
)


class ParseError(Exception):
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"{message} at {line}:{column}")
        self.line = line
        self.column = column


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> Program:
        directives: list[DirectiveNode] = []
        start = self._tokens[0]

        while not self._is_at_end():
            self._skip_newlines()
            if self._is_at_end():
                break
            if self._check(TokenKind.At):
                directives.append(self._parse_directive())
            else:
                self._advance()

        end = self._tokens[min(self._pos, len(self._tokens) - 1)]
        return Program(directives=directives, span=self._make_span(start, end))

    def _parse_directive(self) -> DirectiveNode:
        at_token = self._expect(TokenKind.At)
        name_token = self._expect(TokenKind.Identifier)

        match name_token.value:
            case "model":
                return self._parse_model(at_token)
            case "role":
                return self._parse_role(at_token)
            case "context":
                return self._parse_context(at_token)
            case "task":
                return self._parse_task(at_token)
            case "output":
                return self._parse_output(at_token)
            case "constraint":
                return self._parse_constraint(at_token)
            case "retry":
                return self._parse_retry(at_token)
            case "fn":
                return self._parse_fn(at_token)
            case "import":
                return self._parse_import(at_token)
            case "call":
                return self._parse_call(at_token)
            case _:
                raise ParseError(
                    f"Unknown directive @{name_token.value}",
                    name_token.line,
                    name_token.column,
                )

    def _parse_model(self, at: Token) -> ModelDirective:
        val = self._expect(TokenKind.Text)
        return ModelDirective(value=val.value, span=self._make_span(at, val))

    def _parse_role(self, at: Token) -> RoleDirective:
        value = self._read_inline_or_block()
        return RoleDirective(value=value, span=self._make_span(at, self._previous()))

    def _parse_context(self, at: Token) -> ContextDirective:
        body = self._read_inline_or_block()
        # Check for "file: <path>" syntax
        if body.startswith("file:"):
            file_path = body[len("file:"):].strip()
            return ContextDirective(file=file_path, span=self._make_span(at, self._previous()))
        # Check for "stdin" keyword
        if body.strip() == "stdin":
            return ContextDirective(body="__stdin__", span=self._make_span(at, self._previous()))
        return ContextDirective(body=body, span=self._make_span(at, self._previous()))

    def _parse_task(self, at: Token) -> TaskDirective:
        body = self._read_inline_or_block()
        return TaskDirective(body=body, span=self._make_span(at, self._previous()))

    def _parse_output(self, at: Token) -> OutputDirective:
        props: dict[str, str] = {}
        if self._check(TokenKind.BraceOpen):
            self._advance()
            body = self._expect(TokenKind.Body)
            self._expect(TokenKind.BraceClose)
            self._parse_properties(body.value, props)
        else:
            # Consume all text/comma tokens on this line as key:value pairs
            parts: list[str] = []
            while not self._is_at_end() and not self._check(TokenKind.Newline) and not self._check(TokenKind.At):
                if self._check(TokenKind.Comma):
                    self._advance()
                elif self._check(TokenKind.Text):
                    parts.append(self._advance().value)
                else:
                    break
            self._parse_properties(", ".join(parts), props)
        return OutputDirective(properties=props, span=self._make_span(at, self._previous()))

    def _parse_constraint(self, at: Token) -> ConstraintDirective:
        body = self._read_inline_or_block()
        return ConstraintDirective(body=body, span=self._make_span(at, self._previous()))

    def _parse_retry(self, at: Token) -> RetryDirective:
        text = self._read_inline_or_block().strip()
        try:
            count = int(text)
        except ValueError:
            count = 3
        return RetryDirective(count=count, span=self._make_span(at, self._previous()))

    def _parse_fn(self, at: Token) -> FnDirective:
        name_token = self._expect(TokenKind.Identifier)
        params = self._parse_param_list()
        self._skip_newlines()
        self._expect(TokenKind.BraceOpen)
        body_token = self._expect(TokenKind.Body)
        self._expect(TokenKind.BraceClose)
        return FnDirective(
            name=name_token.value,
            params=params,
            body=body_token.value,
            span=self._make_span(at, self._previous()),
        )

    def _parse_import(self, at: Token) -> ImportDirective:
        path = self._read_inline_or_block()
        return ImportDirective(path=path, span=self._make_span(at, self._previous()))

    def _parse_call(self, at: Token) -> CallDirective:
        name_token = self._expect(TokenKind.Identifier)
        args = self._parse_arg_list()
        return CallDirective(
            name=name_token.value,
            args=args,
            span=self._make_span(at, self._previous()),
        )

    def _parse_param_list(self) -> list[str]:
        if not self._check(TokenKind.ParenOpen):
            return []
        self._advance()
        params: list[str] = []
        while not self._check(TokenKind.ParenClose) and not self._is_at_end():
            self._skip_newlines()
            if self._check(TokenKind.Identifier):
                params.append(self._advance().value)
            elif self._check(TokenKind.Comma):
                self._advance()
            elif self._check(TokenKind.Text):
                text = self._advance().value
                for p in text.split(","):
                    trimmed = p.strip()
                    if trimmed:
                        params.append(trimmed)
            else:
                break
        self._expect(TokenKind.ParenClose)
        return params

    def _parse_arg_list(self) -> list[str]:
        if not self._check(TokenKind.ParenOpen):
            return []
        self._advance()
        args: list[str] = []
        parts: list[str] = []
        while not self._check(TokenKind.ParenClose) and not self._is_at_end():
            if self._check(TokenKind.Comma):
                self._advance()
                args.append(" ".join(parts).strip())
                parts.clear()
            else:
                parts.append(self._advance().value)
        if parts:
            args.append(" ".join(parts).strip())
        self._expect(TokenKind.ParenClose)
        return args

    def _read_inline_or_block(self) -> str:
        if self._check(TokenKind.BraceOpen):
            self._advance()
            body = self._expect(TokenKind.Body)
            self._expect(TokenKind.BraceClose)
            return body.value
        # Consume all tokens on this line as inline text
        parts: list[str] = []
        while (
            not self._is_at_end()
            and not self._check(TokenKind.Newline)
            and not self._check(TokenKind.At)
            and not self._check(TokenKind.BraceOpen)
            and not self._check(TokenKind.EOF)
        ):
            parts.append(self._advance().value)
        return "".join(parts) if parts else ""

    def _parse_properties(self, text: str, out: dict[str, str]) -> None:
        for pair in text.split(","):
            idx = pair.find(":")
            if idx != -1:
                key = pair[:idx].strip()
                value = pair[idx + 1 :].strip()
                if key:
                    out[key] = value

    # --- helpers ---

    def _check(self, kind: TokenKind) -> bool:
        if self._is_at_end():
            return False
        return self._tokens[self._pos].kind == kind

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _expect(self, kind: TokenKind) -> Token:
        if self._check(kind):
            return self._advance()
        current = self._tokens[min(self._pos, len(self._tokens) - 1)]
        raise ParseError(
            f"Expected {kind.value}, got {current.kind.value}",
            current.line,
            current.column,
        )

    def _previous(self) -> Token:
        return self._tokens[self._pos - 1]

    def _is_at_end(self) -> bool:
        return self._pos >= len(self._tokens) or self._tokens[self._pos].kind == TokenKind.EOF

    def _skip_newlines(self) -> None:
        while self._check(TokenKind.Newline):
            self._advance()

    def _make_span(self, start: Token, end: Token) -> Span:
        return Span(
            start=Position(start.line, start.column),
            end=Position(end.line, end.column),
        )
