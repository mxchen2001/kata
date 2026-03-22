"""CLI for the Kata language."""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

from .lexer import Lexer
from .parser import Parser, ParseError
from .compiler import Compiler, Step
from .decompiler import Decompiler
from .diagnostics import diagnose, format_diagnostics, has_errors
from .engine import Engine, load_plan, _load_dotenv


# ── Artifact extraction helpers ──────────────────────────────────────────────

FORMAT_EXT: dict[str, str] = {
    "python": ".py",
    "typescript": ".ts",
    "javascript": ".js",
    "markdown": ".md",
    "json": ".json",
    "yaml": ".yaml",
    "yml": ".yaml",
    "html": ".html",
    "css": ".css",
    "sql": ".sql",
    "shell": ".sh",
    "bash": ".sh",
    "rust": ".rs",
    "go": ".go",
    "java": ".java",
    "ruby": ".rb",
    "text": ".txt",
}

_CODE_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)```\s*$", re.DOTALL)


def _extract_content(text: str) -> str:
    """Strip markdown code fences from LLM output if present."""
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1).rstrip("\n") if m else text


def _artifact_filename(step: Step, index: int) -> str | None:
    """Determine output filename for a step, or None if no file output."""
    if not step.output:
        return None
    # Explicit file name via @output file: name.ext
    if "file" in step.output:
        return step.output["file"]
    # Derive from @output format: <lang>
    fmt = step.output.get("format", "").lower()
    ext = FORMAT_EXT.get(fmt)
    if ext:
        return f"step_{index}{ext}"
    return None


def _emit(content: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(content, encoding="utf-8")
        print(f"Written to {output_path}")
    else:
        print(content)


def _safe_parse(file_path: str):
    try:
        source = Path(file_path).read_text(encoding="utf-8")
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        return source, tokens, ast
    except ParseError as e:
        print(f"{file_path}:{e.line}:{e.column} error: {e}", file=sys.stderr)
        sys.exit(1)


def _check(file_path: str, ast) -> bool:
    ds = diagnose(ast)
    if ds:
        print(format_diagnostics(ds, file_path), file=sys.stderr)
    return has_errors(ds)


def cmd_compile(args: argparse.Namespace) -> None:
    _, _, ast = _safe_parse(args.file)
    if _check(args.file, ast):
        sys.exit(1)

    compiler = Compiler(base_path=Path(args.file).parent)
    plan = compiler.compile(ast)

    if args.output:
        _emit(json.dumps(plan.to_dict(), indent=2), args.output)
    else:
        print(compiler.render(plan))


def cmd_check(args: argparse.Namespace) -> None:
    _, _, ast = _safe_parse(args.file)
    ds = diagnose(ast)
    if ds:
        print(format_diagnostics(ds, args.file), file=sys.stderr)
        sys.exit(1 if has_errors(ds) else 0)
    else:
        print("OK")


def cmd_parse(args: argparse.Namespace) -> None:
    _, _, ast = _safe_parse(args.file)
    import dataclasses
    _emit(json.dumps(dataclasses.asdict(ast), indent=2), args.output)


def _expected_outputs(file_path: str, outdir: Path) -> list[str]:
    """Return list of output filenames a .kata file would produce."""
    if file_path.endswith(".json"):
        return []
    _, _, ast = _safe_parse(file_path)
    plan = Compiler(base_path=outdir).compile(ast)
    filenames: list[str] = []
    for i, step in enumerate(plan.steps):
        fname = _artifact_filename(step, i + 1)
        if fname:
            filenames.append(fname)
    return filenames


def _run_file(file_path: str, outdir: Path, engine: Engine) -> list[str]:
    """Run a single .kata or .json file and write outputs. Returns list of written filenames."""
    if file_path.endswith(".json"):
        plan = load_plan(Path(file_path))
    else:
        _, _, ast = _safe_parse(file_path)
        if _check(file_path, ast):
            sys.exit(1)
        plan = Compiler(base_path=outdir).compile(ast)

    artifact = engine.run(plan)

    if engine._dry_run:
        return []

    outdir.mkdir(parents=True, exist_ok=True)

    # Write full run artifact
    run_path = outdir / "run.json"
    run_path.write_text(
        json.dumps(artifact.to_dict(), indent=2), encoding="utf-8"
    )

    # Extract and write individual artifact files
    written: list[str] = []
    for i, step in enumerate(plan.steps):
        result_text = artifact.outputs.get(step.id)
        if result_text is None:
            continue
        fname = _artifact_filename(step, i + 1)
        if fname:
            content = _extract_content(result_text)
            fpath = outdir / fname
            fpath.write_text(content + "\n", encoding="utf-8")
            written.append(fname)

    # Print results to stdout
    if not engine._stream:
        for r in artifact.results:
            print(f"\n--- {r.step_id} ({r.model}) ---")
            print(r.output)

    # Summary
    print(f"\n{'─' * 40}")
    print(f"Output: {outdir}/")
    print(f"  run.json")
    for w in written:
        print(f"  {w}")

    # Token usage summary
    total = artifact.total_usage
    if total.input_tokens or total.output_tokens:
        print(f"\nTokens: {total.input_tokens} in / {total.output_tokens} out / {total.input_tokens + total.output_tokens} total")

    return written


def cmd_run(args: argparse.Namespace) -> None:
    _load_dotenv()
    file_path: str = args.file
    stem = Path(file_path).stem
    outdir = Path(args.outdir) if args.outdir else Path("output") / stem
    engine = Engine(
        verbose=True,
        dry_run=getattr(args, "dry_run", False),
        stream=getattr(args, "stream", False),
    )
    _run_file(file_path, outdir, engine)


def cmd_exec(args: argparse.Namespace) -> None:
    """Run all .kata files in a directory in sorted order.

    All files share a single output directory so that later files can
    reference outputs from earlier ones via @context file: <path>.
    """
    _load_dotenv()

    source_dir = Path(args.dir)
    if not source_dir.is_dir():
        print(f"Not a directory: {source_dir}", file=sys.stderr)
        sys.exit(1)

    kata_files = sorted(source_dir.glob("*.kata"))
    if not kata_files:
        print(f"No .kata files found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else Path("output") / source_dir.name
    outdir.mkdir(parents=True, exist_ok=True)
    resume = getattr(args, "resume", False)
    engine = Engine(
        verbose=True,
        dry_run=getattr(args, "dry_run", False),
        stream=getattr(args, "stream", False),
    )

    for kata_file in kata_files:
        # --resume: skip files whose outputs already exist
        if resume:
            expected = _expected_outputs(str(kata_file), outdir)
            if expected and all((outdir / f).is_file() for f in expected):
                print(f"\n  {kata_file.name} — skipped (outputs exist)")
                continue

        print(f"\n{'═' * 40}")
        print(f"  {kata_file.name}")
        print(f"{'═' * 40}")
        _run_file(str(kata_file), outdir, engine)


def cmd_decompile(args: argparse.Namespace) -> None:
    data = json.load(sys.stdin)
    decompiler = Decompiler()
    _emit(decompiler.decompile(data), args.output)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kata",
        description="Kata 型 — LLM Programming Language",
    )
    sub = parser.add_subparsers(dest="command")

    # compile
    p_compile = sub.add_parser("compile", aliases=["c"], help="Compile source to execution plan")
    p_compile.add_argument("file", help="Path to .kata file")
    p_compile.add_argument("-o", "--output", help="Write plan as JSON to file")
    p_compile.set_defaults(func=cmd_compile)

    # check
    p_check = sub.add_parser("check", aliases=["k"], help="Validate without compiling")
    p_check.add_argument("file", help="Path to .kata file")
    p_check.set_defaults(func=cmd_check)

    # parse
    p_parse = sub.add_parser("parse", aliases=["p"], help="Parse and dump AST")
    p_parse.add_argument("file", help="Path to .kata file")
    p_parse.add_argument("-o", "--output", help="Write output to file")
    p_parse.set_defaults(func=cmd_parse)

    # run
    p_run = sub.add_parser("run", aliases=["r"], help="Execute a .kata file or compiled .json plan")
    p_run.add_argument("file", help="Path to .kata or .json plan file")
    p_run.add_argument("-d", "--outdir", help="Output directory (default: output/<name>/)")
    p_run.add_argument("--dry-run", action="store_true", help="Show prompts without calling APIs")
    p_run.add_argument("--stream", action="store_true", help="Stream LLM output as it arrives")
    p_run.set_defaults(func=cmd_run)

    # exec
    p_exec = sub.add_parser("exec", aliases=["e"], help="Run all .kata files in a directory in order")
    p_exec.add_argument("dir", help="Directory containing .kata files")
    p_exec.add_argument("-d", "--outdir", help="Shared output directory (default: output/<dirname>/)")
    p_exec.add_argument("--dry-run", action="store_true", help="Show prompts without calling APIs")
    p_exec.add_argument("--stream", action="store_true", help="Stream LLM output as it arrives")
    p_exec.add_argument("--resume", action="store_true", help="Skip files whose outputs already exist")
    p_exec.set_defaults(func=cmd_exec)

    # decompile
    p_decompile = sub.add_parser("decompile", aliases=["d"], help="Read plan JSON from stdin, output .kata source")
    p_decompile.add_argument("-o", "--output", help="Write output to file")
    p_decompile.set_defaults(func=cmd_decompile)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)
