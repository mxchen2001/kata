"""Kata 型 — a structured programming language targeting LLM inference."""

from .ast import *
from .lexer import Lexer, Token, TokenKind
from .parser import Parser, ParseError
from .compiler import Compiler, ExecutionPlan, Step
from .decompiler import Decompiler
from .engine import Engine, StepResult, RunArtifact, load_plan
from .diagnostics import diagnose, format_diagnostics, has_errors, Diagnostic, Severity
