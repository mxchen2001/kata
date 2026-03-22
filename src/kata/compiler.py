"""Compiler for the Kata language — AST → execution plan."""

from __future__ import annotations
from dataclasses import dataclass, field
from .ast import Program, DirectiveNode, FnDirective
from .lexer import Lexer
from .parser import Parser


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

    def compile(self, program: Program) -> ExecutionPlan:
        fns: dict[str, FnDirective] = {}
        for d in program.directives:
            if d.kind == "fn":
                assert isinstance(d, FnDirective)
                fns[d.name] = d

        expanded = self._expand_calls(program.directives, fns)
        plan = self._build_plan(expanded)
        plan = self._optimize(plan)
        return plan

    def _expand_calls(
        self,
        directives: list[DirectiveNode],
        fns: dict[str, FnDirective],
    ) -> list[DirectiveNode]:
        result: list[DirectiveNode] = []

        for d in directives:
            if d.kind == "fn":
                continue
            if d.kind == "call":
                fn = fns.get(d.name)  # type: ignore[union-attr]
                if fn is None:
                    raise RuntimeError(
                        f"Undefined function @{d.name} at {d.span.start.line}:{d.span.start.column}"  # type: ignore[union-attr]
                    )
                body = fn.body
                for i, param in enumerate(fn.params):
                    placeholder = f"${{{param}}}"
                    value = d.args[i] if i < len(d.args) else ""  # type: ignore[union-attr]
                    body = body.replace(placeholder, value)
                inner_tokens = Lexer(body).tokenize()
                inner_ast = Parser(inner_tokens).parse()
                inner_expanded = self._expand_calls(inner_ast.directives, fns)
                result.extend(inner_expanded)
            else:
                result.append(d)

        return result

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

        for d in directives:
            match d.kind:
                case "model":
                    model = d.value  # type: ignore[union-attr]
                case "role":
                    system_parts.append(d.value)  # type: ignore[union-attr]
                case "context":
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
