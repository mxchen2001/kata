"""Semantic diagnostics for the Kata language."""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal
from .ast import Program, DirectiveNode, Span

Severity = Literal["error", "warning"]

_SINGULAR_DIRECTIVES = {"model", "role"}


@dataclass
class Diagnostic:
    severity: Severity
    message: str
    span: Span


def diagnose(program: Program) -> list[Diagnostic]:
    ds: list[Diagnostic] = []

    if not program.directives:
        ds.append(Diagnostic("error", "Empty program — no directives found", program.span))
        return ds

    # Count directives by kind
    counts: dict[str, list[DirectiveNode]] = {}
    for d in program.directives:
        counts.setdefault(d.kind, []).append(d)

    # Duplicate singular directives
    for kind in _SINGULAR_DIRECTIVES:
        nodes = counts.get(kind, [])
        if len(nodes) > 1:
            for d in nodes[1:]:
                ds.append(Diagnostic(
                    "error",
                    f"Duplicate @{kind} — must be declared exactly once",
                    d.span,
                ))

    # Required: @model (unless @call directives provide one)
    if "model" not in counts and "call" not in counts:
        ds.append(Diagnostic(
            "error",
            "Missing @model — every program must specify a target model",
            program.span,
        ))

    # Required: @task (unless @call directives provide one)
    if "task" not in counts and "call" not in counts:
        ds.append(Diagnostic(
            "error",
            "Missing @task — every program must include a task",
            program.span,
        ))

    # Empty bodies
    for d in program.directives:
        if d.kind == "role" and not d.value.strip():  # type: ignore[union-attr]
            ds.append(Diagnostic("warning", "@role has an empty body", d.span))
        if d.kind == "context" and not d.body.strip():  # type: ignore[union-attr]
            ds.append(Diagnostic("warning", "@context has an empty body", d.span))
        if d.kind == "task" and not d.body.strip():  # type: ignore[union-attr]
            ds.append(Diagnostic("warning", "@task has an empty body", d.span))
        if d.kind == "constraint" and not d.body.strip():  # type: ignore[union-attr]
            ds.append(Diagnostic("warning", "@constraint has an empty body", d.span))
        if d.kind == "output":
            if not d.properties:  # type: ignore[union-attr]
                ds.append(Diagnostic(
                    "warning",
                    "@output has no properties — expected key: value pairs",
                    d.span,
                ))

    # @constraint without @task
    if "constraint" in counts and "task" not in counts:
        ds.append(Diagnostic(
            "warning",
            "@constraint without @task — constraints will have no effect",
            counts["constraint"][0].span,
        ))

    # Nested directive leak (skip @fn bodies which are expected to have directives)
    for d in program.directives:
        if d.kind == "fn":
            continue
        body = _get_body(d)
        if body is None:
            continue
        m = re.search(r"^[ \t]*@(\w+)", body, re.MULTILINE)
        if m:
            ds.append(Diagnostic(
                "warning",
                f"Block body contains @{m.group(1)} — nested directives are treated as plain text, not parsed",
                d.span,
            ))

    # --- Function diagnostics ---

    fn_defs: dict[str, list[DirectiveNode]] = {}
    for d in program.directives:
        if d.kind == "fn":
            fn_defs.setdefault(d.name, []).append(d)  # type: ignore[union-attr]

    # Duplicate function names
    for name, nodes in fn_defs.items():
        if len(nodes) > 1:
            for d in nodes[1:]:
                ds.append(Diagnostic(
                    "error",
                    f'Duplicate @fn "{name}" — function names must be unique',
                    d.span,
                ))

    # Validate @call references
    for d in program.directives:
        if d.kind == "call":
            fn_list = fn_defs.get(d.name)  # type: ignore[union-attr]
            if fn_list is None:
                ds.append(Diagnostic(
                    "error",
                    f'@call "{d.name}" — no @fn with that name is defined',  # type: ignore[union-attr]
                    d.span,
                ))
            else:
                fn_def = fn_list[0]
                if fn_def.kind == "fn":
                    expected = len(fn_def.params)  # type: ignore[union-attr]
                    got = len(d.args)  # type: ignore[union-attr]
                    if got != expected:
                        s = "" if expected == 1 else "s"
                        ds.append(Diagnostic(
                            "error",
                            f'@call "{d.name}" — expected {expected} argument{s}, got {got}',  # type: ignore[union-attr]
                            d.span,
                        ))

    # Unused functions
    called_names = {d.name for d in program.directives if d.kind == "call"}  # type: ignore[union-attr]
    for name, nodes in fn_defs.items():
        if name not in called_names:
            ds.append(Diagnostic(
                "warning",
                f'@fn "{name}" is defined but never called',
                nodes[0].span,
            ))

    return ds


def _get_body(d: DirectiveNode) -> str | None:
    match d.kind:
        case "role":
            return d.value  # type: ignore[union-attr]
        case "context" | "task" | "constraint":
            return d.body  # type: ignore[union-attr]
        case _:
            return None


def format_diagnostics(ds: list[Diagnostic], file_path: str | None = None) -> str:
    prefix = f"{file_path}:" if file_path else ""
    lines: list[str] = []
    for d in ds:
        loc = f"{prefix}{d.span.start.line}:{d.span.start.column}"
        if d.severity == "error":
            tag = "\033[31merror\033[0m"
        else:
            tag = "\033[33mwarning\033[0m"
        lines.append(f"{loc} {tag}: {d.message}")
    return "\n".join(lines)


def has_errors(ds: list[Diagnostic]) -> bool:
    return any(d.severity == "error" for d in ds)
