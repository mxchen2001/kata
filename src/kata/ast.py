"""AST node types for the Kata language."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union


@dataclass
class Position:
    line: int
    column: int


@dataclass
class Span:
    start: Position
    end: Position


@dataclass
class ModelDirective:
    kind: str = field(default="model", init=False)
    value: str = ""
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class RoleDirective:
    kind: str = field(default="role", init=False)
    value: str = ""
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class ContextDirective:
    kind: str = field(default="context", init=False)
    body: str = ""
    file: str | None = None  # @context file: <path> — inline file contents at compile time
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class TaskDirective:
    kind: str = field(default="task", init=False)
    body: str = ""
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class OutputDirective:
    kind: str = field(default="output", init=False)
    properties: dict[str, str] = field(default_factory=dict)
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class ConstraintDirective:
    kind: str = field(default="constraint", init=False)
    body: str = ""
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


# TODO: ChainStep/ChainDirective are AST-only — not yet wired into lexer, parser, or compiler.
@dataclass
class ChainStep:
    name: str = ""
    directives: list[DirectiveNode] = field(default_factory=list)
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class ChainDirective:
    kind: str = field(default="chain", init=False)
    steps: list[ChainStep] = field(default_factory=list)
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class FnDirective:
    kind: str = field(default="fn", init=False)
    name: str = ""
    params: list[str] = field(default_factory=list)
    body: str = ""
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class RetryDirective:
    kind: str = field(default="retry", init=False)
    count: int = 3
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class ImportDirective:
    kind: str = field(default="import", init=False)
    path: str = ""
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


@dataclass
class CallDirective:
    kind: str = field(default="call", init=False)
    name: str = ""
    args: list[str] = field(default_factory=list)
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))


DirectiveNode = Union[
    ModelDirective,
    RoleDirective,
    ContextDirective,
    TaskDirective,
    OutputDirective,
    ConstraintDirective,
    ChainDirective,
    FnDirective,
    RetryDirective,
    ImportDirective,
    CallDirective,
]


@dataclass
class Program:
    directives: list[DirectiveNode] = field(default_factory=list)
    span: Span = field(default_factory=lambda: Span(Position(0, 0), Position(0, 0)))
