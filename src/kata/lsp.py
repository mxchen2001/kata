"""Language Server Protocol implementation for Kata."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .lexer import Lexer
from .parser import Parser, ParseError
from .compiler import Compiler
from .diagnostics import diagnose, Diagnostic
from .roles import list_presets, expand_role

_ROLE_PRESET_RE = re.compile(r":(\w+)(?:\(([^)]*)\))?")


def _parse_source(source: str):
    """Parse source, returning (ast, error_diagnostic) tuple."""
    try:
        source = Compiler.preprocess(source)
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        return ast, None
    except ParseError as e:
        return None, e


def _get_diagnostics(source: str, uri: str) -> list[dict]:
    """Run diagnostics and return LSP-format diagnostic dicts."""
    ast, parse_error = _parse_source(source)
    results: list[dict] = []

    if parse_error:
        results.append({
            "range": {
                "start": {"line": parse_error.line - 1, "character": parse_error.column - 1},
                "end": {"line": parse_error.line - 1, "character": parse_error.column + 10},
            },
            "severity": 1,  # Error
            "source": "kata",
            "message": str(parse_error),
        })
        return results

    if ast is None:
        return results

    for d in diagnose(ast):
        severity = 1 if d.severity == "error" else 2  # 1=Error, 2=Warning
        results.append({
            "range": {
                "start": {"line": d.span.start.line - 1, "character": d.span.start.column - 1},
                "end": {"line": d.span.end.line - 1, "character": d.span.end.column},
            },
            "severity": severity,
            "source": "kata",
            "message": d.message,
        })

    return results


def _get_hover(source: str, line: int, character: int) -> str | None:
    """Return hover text for position, or None."""
    lines = source.splitlines()
    if line >= len(lines):
        return None

    text = lines[line]

    # Check for role preset hover
    m = re.search(r"@role\s+(:\w+(?:\([^)]*\))?)", text)
    if m and m.start(1) <= character <= m.end(1):
        try:
            expanded = expand_role(m.group(1))
            return f"**Role preset** `{m.group(1)}`\n\nExpands to: *{expanded}*"
        except ValueError as e:
            return str(e)

    # Check for directive hover
    m = re.search(r"@(\w+)", text)
    if m and m.start() <= character <= m.end():
        directive = m.group(1)
        docs = {
            "model": "Target LLM model (e.g. `gpt-4o`, `claude-sonnet-4-6`)",
            "role": "System prompt — use `:preset` syntax or free-form text",
            "context": "Additional context. `file: path`, `file: *.py` (glob), or `stdin`",
            "task": "The user prompt — what you want the model to do",
            "constraint": "A rule the model must follow",
            "output": "Output format and filename (`format: python, file: out.py`)",
            "fn": "Define a reusable function with parameters",
            "call": "Invoke a defined `@fn`",
            "import": "Import `@fn` definitions from another `.kata` file",
            "retry": "Retry a step N times on failure",
            "chain": "Define a multi-step pipeline with named `@step` blocks",
            "var": "Define a variable for `${name}` substitution",
            "if": "Conditionally include directives (`file_exists(path)` or `env(VAR)`)",
        }
        if directive in docs:
            return f"**@{directive}**\n\n{docs[directive]}"

    return None


def _get_definitions(source: str, line: int, character: int) -> list[dict] | None:
    """Return go-to-definition locations for @call -> @fn."""
    lines = source.splitlines()
    if line >= len(lines):
        return None

    text = lines[line]

    # Check if cursor is on a @call name
    m = re.search(r"@call\s+(\w+)", text)
    if not m or not (m.start(1) <= character <= m.end(1)):
        return None

    fn_name = m.group(1)

    # Find the @fn definition
    for i, ln in enumerate(lines):
        fn_match = re.search(rf"@fn\s+{re.escape(fn_name)}\b", ln)
        if fn_match:
            return [{
                "line": i,
                "character": fn_match.start(),
            }]

    return None


def main() -> None:
    """Run the Kata LSP server over stdin/stdout."""
    try:
        from pygls.server import LanguageServer
        from lsprotocol import types
    except ImportError:
        print(
            "LSP requires pygls: pip install 'pygls>=1.0' lsprotocol",
            file=sys.stderr,
        )
        sys.exit(1)

    server = LanguageServer("kata-lsp", "0.1.0")

    def _publish(ls: LanguageServer, uri: str, source: str) -> None:
        diagnostics = _get_diagnostics(source, uri)
        ls.publish_diagnostics(
            uri,
            [types.Diagnostic(
                range=types.Range(
                    start=types.Position(line=d["range"]["start"]["line"], character=d["range"]["start"]["character"]),
                    end=types.Position(line=d["range"]["end"]["line"], character=d["range"]["end"]["character"]),
                ),
                severity=types.DiagnosticSeverity(d["severity"]),
                source=d["source"],
                message=d["message"],
            ) for d in diagnostics],
        )

    @server.feature(types.TEXT_DOCUMENT_DID_OPEN)
    def did_open(params: types.DidOpenTextDocumentParams) -> None:
        _publish(server, params.text_document.uri, params.text_document.text)

    @server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
    def did_change(params: types.DidChangeTextDocumentParams) -> None:
        doc = server.workspace.get_text_document(params.text_document.uri)
        _publish(server, params.text_document.uri, doc.source)

    @server.feature(types.TEXT_DOCUMENT_DID_SAVE)
    def did_save(params: types.DidSaveTextDocumentParams) -> None:
        doc = server.workspace.get_text_document(params.text_document.uri)
        _publish(server, params.text_document.uri, doc.source)

    @server.feature(types.TEXT_DOCUMENT_HOVER)
    def hover(params: types.HoverParams) -> types.Hover | None:
        doc = server.workspace.get_text_document(params.text_document.uri)
        result = _get_hover(doc.source, params.position.line, params.position.character)
        if result:
            return types.Hover(
                contents=types.MarkupContent(kind=types.MarkupKind.Markdown, value=result),
            )
        return None

    @server.feature(types.TEXT_DOCUMENT_DEFINITION)
    def definition(params: types.DefinitionParams) -> list[types.Location] | None:
        doc = server.workspace.get_text_document(params.text_document.uri)
        defs = _get_definitions(doc.source, params.position.line, params.position.character)
        if defs:
            return [
                types.Location(
                    uri=params.text_document.uri,
                    range=types.Range(
                        start=types.Position(line=d["line"], character=d["character"]),
                        end=types.Position(line=d["line"], character=d["character"] + 1),
                    ),
                )
                for d in defs
            ]
        return None

    server.start_io()


if __name__ == "__main__":
    main()
