"""Compiler for the Kata language — AST → execution plan."""

from __future__ import annotations
import glob as globmod
import re
from dataclasses import dataclass, field
from pathlib import Path
from .ast import Program, DirectiveNode, FnDirective, ImportDirective, TaskDirective
from .lexer import Lexer
from .parser import Parser
from .roles import expand_role

_INLINE_USE_RE = re.compile(r"@use\s+(\w+)(?:\(([^)]*)\))?")


@dataclass
class Step:
    """A single step in an execution plan — one LLM call."""
    id: str
    model: str | None = None
    system: str | None = None
    user: str = ""
    constraints: list[str] = field(default_factory=list)
    output: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    retries: int = 0

    def to_dict(self) -> dict:
        d: dict = {"id": self.id}
        if self.model:
            d["model"] = self.model
        if self.system:
            d["system"] = self.system
        d["user"] = self.user
        if self.constraints:
            d["constraints"] = self.constraints
        if self.output:
            d["output"] = self.output
        if self.depends_on:
            d["depends_on"] = self.depends_on
        if self.retries:
            d["retries"] = self.retries
        return d


@dataclass
class ExecutionPlan:
    """The compiled output of a Kata program — a sequence of steps for an LLM engine."""
    steps: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan": [s.to_dict() for s in self.steps],
        }


class Compiler:
    """Compiles a Kata AST into an execution plan.

    Each top-level prompt or @call expansion becomes a Step.
    Steps with different @model values are separate steps.
    Sequential calls create dependency chains.
    """

    def __init__(self, *, base_path: Path | None = None) -> None:
        self._base_path = base_path or Path(".")

    def compile(self, program: Program) -> ExecutionPlan:
        fns: dict[str, FnDirective] = {}

        # Resolve @import directives — pull in @fn defs from other files
        for d in program.directives:
            if d.kind == "import":
                assert isinstance(d, ImportDirective)
                imported_fns = self._resolve_import(d.path)
                for fn in imported_fns:
                    fns[fn.name] = fn

        for d in program.directives:
            if d.kind == "fn":
                assert isinstance(d, FnDirective)
                fns[d.name] = d

        expanded = self._expand_calls(program.directives, fns)
        ref_steps, expanded = self._resolve_inline_uses(expanded, fns)
        plan = self._build_plan(expanded)

        if ref_steps:
            # Wire dependencies: main steps that use {{ref:...}} depend on ref steps
            for step in plan.steps:
                for ref in ref_steps:
                    marker = f"{{{{ref:{ref.id}}}}}"
                    if marker in step.user:
                        if ref.id not in step.depends_on:
                            step.depends_on.append(ref.id)
            plan.steps = ref_steps + plan.steps

        plan = self._optimize(plan)
        return plan

    def _resolve_import(self, path: str) -> list[FnDirective]:
        """Load a .kata file and extract its @fn definitions."""
        file_path = self._base_path / path
        if not file_path.suffix:
            file_path = file_path.with_suffix(".kata")
        source = file_path.read_text(encoding="utf-8")
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        return [d for d in ast.directives if d.kind == "fn"]

    def _expand_calls(
        self,
        directives: list[DirectiveNode],
        fns: dict[str, FnDirective],
        _expanding: frozenset[str] | None = None,
    ) -> list[DirectiveNode]:
        if _expanding is None:
            _expanding = frozenset()

        result: list[DirectiveNode] = []

        for d in directives:
            if d.kind == "fn":
                continue
            if d.kind == "call":
                fn_name: str = d.name  # type: ignore[union-attr]
                if fn_name in _expanding:
                    raise RuntimeError(
                        f"Recursive call cycle detected: @{fn_name} "
                        f"at {d.span.start.line}:{d.span.start.column}"  # type: ignore[union-attr]
                    )
                fn = fns.get(fn_name)
                if fn is None:
                    raise RuntimeError(
                        f"Undefined function @{fn_name} at {d.span.start.line}:{d.span.start.column}"  # type: ignore[union-attr]
                    )
                body = fn.body
                for i, param in enumerate(fn.params):
                    placeholder = f"${{{param}}}"
                    value = d.args[i] if i < len(d.args) else ""  # type: ignore[union-attr]
                    body = body.replace(placeholder, value)
                inner_tokens = Lexer(body).tokenize()
                inner_ast = Parser(inner_tokens).parse()
                inner_expanded = self._expand_calls(
                    inner_ast.directives, fns, _expanding | {fn_name}
                )
                result.extend(inner_expanded)
            else:
                result.append(d)

        return result

    def _resolve_inline_uses(
        self,
        directives: list[DirectiveNode],
        fns: dict[str, FnDirective],
    ) -> tuple[list[Step], list[DirectiveNode]]:
        """Resolve @use references inside @task bodies.

        Each @use is compiled into ref steps that run first.
        The @use text is replaced with {{ref:ref_N}} which the engine
        resolves at runtime with the step's actual output.

        Syntax: @call at top level creates a sibling step in the chain.
                @use inside @task inlines the resolved output in-place.
        """
        ref_steps: list[Step] = []
        modified: list[DirectiveNode] = []

        for d in directives:
            if d.kind == "task" and "@use" in d.body:  # type: ignore[union-attr]
                new_body: str = d.body  # type: ignore[union-attr]
                for match in _INLINE_USE_RE.finditer(new_body):
                    fn_name = match.group(1)
                    args_str = match.group(2) or ""
                    args = [a.strip() for a in args_str.split(",") if a.strip()] if args_str.strip() else []

                    fn = fns.get(fn_name)
                    if fn is None:
                        raise RuntimeError(f"Undefined function @{fn_name} in @use")

                    # Expand function body with args
                    fn_body = fn.body
                    for i, param in enumerate(fn.params):
                        fn_body = fn_body.replace(f"${{{param}}}", args[i] if i < len(args) else "")

                    # Compile into steps
                    inner_tokens = Lexer(fn_body).tokenize()
                    inner_ast = Parser(inner_tokens).parse()
                    inner_expanded = self._expand_calls(inner_ast.directives, fns)

                    inner_segments = self._segment_by_model(inner_expanded)
                    for j, seg in enumerate(inner_segments):
                        step_id = f"ref_{len(ref_steps) + 1}"
                        step = self._build_step(step_id, seg)
                        if j > 0:
                            step.depends_on = [ref_steps[-1].id]
                        ref_steps.append(step)

                    last_ref_id = ref_steps[-1].id
                    new_body = new_body.replace(match.group(0), f"{{{{ref:{last_ref_id}}}}}")

                modified.append(TaskDirective(body=new_body, span=d.span))
            else:
                modified.append(d)

        return ref_steps, modified

    def _build_plan(self, directives: list[DirectiveNode]) -> ExecutionPlan:
        """Build an execution plan from expanded directives.

        Each @model boundary starts a new step. Steps are chained
        sequentially via depends_on.
        """
        segments = self._segment_by_model(directives)
        steps: list[Step] = []

        for i, seg in enumerate(segments):
            step = self._build_step(f"step_{i + 1}", seg)
            if i > 0:
                step.depends_on = [steps[i - 1].id]
            steps.append(step)

        return ExecutionPlan(steps=steps)

    def _segment_by_model(self, directives: list[DirectiveNode]) -> list[list[DirectiveNode]]:
        """Split directives into segments by @model boundary.

        Shared directives (before any @model) are prepended to every segment.
        """
        segments: list[list[DirectiveNode]] = []
        shared: list[DirectiveNode] = []
        current: list[DirectiveNode] | None = None

        for d in directives:
            if d.kind == "model":
                current = [*shared, d]
                segments.append(current)
            elif current is None:
                shared.append(d)
            else:
                current.append(d)

        if not segments:
            return [shared]

        return segments

    def _build_step(self, step_id: str, directives: list[DirectiveNode]) -> Step:
        model: str | None = None
        system_parts: list[str] = []
        user_parts: list[str] = []
        constraints: list[str] = []
        output: dict[str, str] = {}
        retries: int = 0

        for d in directives:
            match d.kind:
                case "model":
                    model = d.value  # type: ignore[union-attr]
                case "retry":
                    retries = d.count  # type: ignore[union-attr]
                case "role":
                    system_parts.append(expand_role(d.value))  # type: ignore[union-attr]
                case "context":
                    if d.file:  # type: ignore[union-attr]
                        pattern = d.file  # type: ignore[union-attr]
                        if any(c in pattern for c in ("*", "?", "[")):
                            # Glob pattern
                            matches = sorted(globmod.glob(str(self._base_path / pattern)))
                            for match in matches:
                                p = Path(match)
                                if p.is_file():
                                    content = p.read_text(encoding="utf-8")
                                    system_parts.append(f"[{p.name}]\n{content}")
                            if not matches:
                                system_parts.append(f"[{pattern} — no files matched]")
                        else:
                            file_path = self._base_path / pattern
                            if file_path.is_file():
                                content = file_path.read_text(encoding="utf-8")
                                system_parts.append(f"[{pattern}]\n{content}")
                            else:
                                system_parts.append(f"[{pattern} — file not found]")
                    elif d.body == "__stdin__":  # type: ignore[union-attr]
                        import sys as _sys
                        if not _sys.stdin.isatty():
                            system_parts.append(_sys.stdin.read())
                    else:
                        system_parts.append(d.body)  # type: ignore[union-attr]
                case "task":
                    user_parts.append(d.body)  # type: ignore[union-attr]
                case "constraint":
                    constraints.append(d.body)  # type: ignore[union-attr]
                case "output":
                    output.update(d.properties)  # type: ignore[union-attr]

        return Step(
            id=step_id,
            model=model,
            system="\n\n".join(system_parts) if system_parts else None,
            user="\n\n".join(user_parts),
            constraints=constraints,
            output=output,
            retries=retries,
        )

    # -- Optimization passes --------------------------------------------------

    def _optimize(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Run optimization passes on the execution plan."""
        plan = self._remove_empty_steps(plan)
        return plan

    def _remove_empty_steps(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Remove steps that produce no LLM call.

        A step is empty when it has no system prompt, no user content,
        no constraints, and no output config (e.g. a bare @model with
        nothing else).  Dependencies are rewired past removed steps and
        IDs are renumbered.
        """
        empty_ids: set[str] = set()
        deps_of: dict[str, list[str]] = {}

        for step in plan.steps:
            deps_of[step.id] = step.depends_on
            if not step.system and not step.user and not step.constraints and not step.output:
                empty_ids.add(step.id)

        if not empty_ids:
            return plan

        # Keep non-empty steps, rewiring deps past removed ones
        kept: list[Step] = []
        for step in plan.steps:
            if step.id in empty_ids:
                continue
            new_deps: list[str] = []
            for dep in step.depends_on:
                if dep in empty_ids:
                    new_deps.extend(deps_of[dep])
                else:
                    new_deps.append(dep)
            step.depends_on = new_deps
            kept.append(step)

        # Renumber step IDs and update references
        id_map: dict[str, str] = {}
        for i, step in enumerate(kept):
            new_id = f"step_{i + 1}"
            id_map[step.id] = new_id
            step.id = new_id

        for step in kept:
            step.depends_on = [id_map[d] for d in step.depends_on]

        return ExecutionPlan(steps=kept)

    def render(self, plan: ExecutionPlan) -> str:
        lines: list[str] = []
        for step in plan.steps:
            header = f"[{step.id}]"
            if step.model:
                header += f" model: {step.model}"
            if step.depends_on:
                header += f" \u2190 {', '.join(step.depends_on)}"
            lines.append(header)

            if step.system:
                lines.append(f"  system: {step.system}")
            lines.append(f"  user: {step.user}")

            if step.constraints:
                lines.append("  constraints:")
                for c in step.constraints:
                    lines.append(f"    - {c}")

            if step.output:
                parts = ", ".join(f"{k}: {v}" for k, v in step.output.items())
                lines.append(f"  output: {parts}")

            lines.append("")

        return "\n".join(lines).rstrip()
