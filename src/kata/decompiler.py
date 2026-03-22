"""Decompiler for the Kata language — execution plan → source (best-effort)."""

from __future__ import annotations
from .compiler import ExecutionPlan, Step


class Decompiler:
    def decompile(self, data: ExecutionPlan | dict) -> str:
        if isinstance(data, dict):
            steps_data = data.get("plan", [data])
            steps = [
                Step(
                    id=s.get("id", f"step_{i+1}"),
                    model=s.get("model"),
                    system=s.get("system"),
                    user=s.get("user", ""),
                    constraints=s.get("constraints", []),
                    output=s.get("output", {}),
                    depends_on=s.get("depends_on", []),
                )
                for i, s in enumerate(steps_data)
            ]
            data = ExecutionPlan(steps=steps)

        if len(data.steps) == 1:
            return self._decompile_step(data.steps[0])

        # Multi-step → wrap each in @fn and emit @call sequence
        parts: list[str] = []
        for step in data.steps:
            fn_body = self._step_to_directives(step)
            indented = fn_body.replace("\n", "\n  ")
            parts.append(f"@fn {step.id} {{\n  {indented}\n}}")

        for step in data.steps:
            parts.append(f"@call {step.id}")

        return "\n\n".join(parts)

    def _decompile_step(self, step: Step) -> str:
        lines: list[str] = []

        if step.model:
            lines.append(f"@model {step.model}")

        if step.system:
            parts = step.system.split("\n\n")
            lines.append(f"@role {parts[0]}")
            if len(parts) > 1:
                context_body = "\n\n".join(parts[1:])
                indented = context_body.replace("\n", "\n  ")
                lines.append(f"@context {{\n  {indented}\n}}")

        if step.user:
            if "\n" in step.user:
                indented = step.user.replace("\n", "\n  ")
                lines.append(f"@task {{\n  {indented}\n}}")
            else:
                lines.append(f"@task {step.user}")

        for c in step.constraints:
            lines.append(f"@constraint {c}")

        if step.output:
            pairs = ", ".join(f"{k}: {v}" for k, v in step.output.items())
            lines.append(f"@output {pairs}")

        return "\n\n".join(lines)

    def _step_to_directives(self, step: Step) -> str:
        """Convert a step to raw directive text for embedding in @fn body."""
        lines: list[str] = []
        if step.model:
            lines.append(f"@model {step.model}")
        if step.system:
            lines.append(f"@role {step.system}")
        if step.user:
            lines.append(f"@task {step.user}")
        for c in step.constraints:
            lines.append(f"@constraint {c}")
        if step.output:
            pairs = ", ".join(f"{k}: {v}" for k, v in step.output.items())
            lines.append(f"@output {pairs}")
        return "\n".join(lines)
