"""Lexer for the Kata language — source → tokens."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import re


class TokenKind(Enum):
    At = "At"
    Identifier = "Identifier"
    BraceOpen = "BraceOpen"
    BraceClose = "BraceClose"
    Text = "Text"
    Body = "Body"
    ParenOpen = "ParenOpen"
    ParenClose = "ParenClose"
    Colon = "Colon"
    Comma = "Comma"
    Newline = "Newline"
    EOF = "EOF"


@dataclass
class Token:
    kind: TokenKind
    value: str
    line: int
    column: int


_IDENT_RE = re.compile(r"[a-zA-Z_]")
_IDENT_CONT_RE = re.compile(r"[a-zA-Z_0-9]")
_WS_RE = re.compile(r"[ \t\r]")


class Lexer:
    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._line = 1
        self._column = 1
        self._tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        while self._pos < len(self._source):
            self._skip_ws()
            if self._pos >= len(self._source):
                break

            ch = self._source[self._pos]

            if ch == "\n":
                self._emit(TokenKind.Newline, "\n")
                self._advance()
                continue

            if ch == "#":
                self._skip_line_comment()
                continue

            if ch == "@":
                self._emit(TokenKind.At, "@")
                self._advance()
                self._read_directive_name()
                continue

            if ch == "(":
                self._emit(TokenKind.ParenOpen, "(")
                self._advance()
                continue

            if ch == ")":
                self._emit(TokenKind.ParenClose, ")")
                self._advance()
                continue

            if ch == "{":
                self._emit(TokenKind.BraceOpen, "{")
                self._advance()
                self._read_block_body()
                continue

            if ch == ":":
                self._emit(TokenKind.Colon, ":")
                self._advance()
                continue

            if ch == ",":
                self._emit(TokenKind.Comma, ",")
                self._advance()
                continue

            self._read_inline_text()

        self._emit(TokenKind.EOF, "")
        return self._tokens

    def _read_directive_name(self) -> None:
        start = self._pos
        while self._pos < len(self._source) and _IDENT_RE.match(self._source[self._pos]):
            self._advance()
        if self._pos > start:
            name = self._source[start : self._pos]
            self._emit(TokenKind.Identifier, name)

            if name in ("fn", "call"):
                self._skip_ws()
                name_start = self._pos
                while self._pos < len(self._source) and _IDENT_CONT_RE.match(self._source[self._pos]):
                    self._advance()
                if self._pos > name_start:
                    self._emit(TokenKind.Identifier, self._source[name_start : self._pos])

    def _read_inline_text(self) -> None:
        start = self._pos
        while (
            self._pos < len(self._source)
            and self._source[self._pos] not in ("\n", "{", "(", ")", ",")
        ):
            self._advance()
        text = self._source[start : self._pos].strip()
        if text:
            self._emit(TokenKind.Text, text)

    def _read_block_body(self) -> None:
        depth = 1
        start = self._pos
        while self._pos < len(self._source) and depth > 0:
            if self._source[self._pos] == "{":
                depth += 1
            elif self._source[self._pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            self._advance()
        body = self._source[start : self._pos].strip()
        self._emit(TokenKind.Body, body)

        if self._pos < len(self._source) and self._source[self._pos] == "}":
            self._emit(TokenKind.BraceClose, "}")
            self._advance()

    def _skip_line_comment(self) -> None:
        while self._pos < len(self._source) and self._source[self._pos] != "\n":
            self._advance()

    def _skip_ws(self) -> None:
        while self._pos < len(self._source) and _WS_RE.match(self._source[self._pos]):
            self._advance()

    def _advance(self) -> None:
        if self._source[self._pos] == "\n":
            self._line += 1
            self._column = 1
        else:
            self._column += 1
        self._pos += 1

    def _emit(self, kind: TokenKind, value: str) -> None:
        self._tokens.append(Token(kind, value, self._line, self._column))
