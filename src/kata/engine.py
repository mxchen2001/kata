"""Runtime engine for executing Kata execution plans."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .compiler import ExecutionPlan, Step


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_id: str
    model: str
    output: str

    def to_dict(self) -> dict:
        return {"step_id": self.step_id, "model": self.model, "output": self.output}


@dataclass
class RunArtifact:
    """Complete record of a plan execution — plan + LLM outputs.

    Serializes as a standard plan with a ``result`` field on each step,
    so ``kata decompile`` can round-trip it back to ``.kata`` source.
    """
    plan: ExecutionPlan
    outputs: dict[str, str]  # step_id → LLM response text

    def to_dict(self) -> dict:
        steps = []
        for step in self.plan.steps:
            d = step.to_dict()
            if step.id in self.outputs:
                d["result"] = self.outputs[step.id]
            steps.append(d)
        return {"plan": steps}

    @property
    def results(self) -> list[StepResult]:
        return [
            StepResult(step.id, step.model or "unknown", self.outputs.get(step.id, ""))
            for step in self.plan.steps
        ]


def load_plan(source: dict | str | Path) -> ExecutionPlan:
    """Load an ExecutionPlan from a JSON dict, string, or file path.

    Supports both the standard {"plan": [...]} format and
    a single-step dict with model/system/user fields.
    """
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        data = source

    if "plan" in data:
        steps = []
        for s in data["plan"]:
            steps.append(Step(
                id=s["id"],
                model=s.get("model"),
                system=s.get("system"),
                user=s.get("user", ""),
                constraints=s.get("constraints", []),
                output=s.get("output", {}),
                depends_on=s.get("depends_on", []),
            ))
        return ExecutionPlan(steps=steps)

    # Single-step plan (no "plan" wrapper)
    return ExecutionPlan(steps=[Step(
        id="step_1",
        model=data.get("model"),
        system=data.get("system"),
        user=data.get("user", ""),
        constraints=data.get("constraints", []),
        output=data.get("output", {}),
    )])


def _load_dotenv() -> None:
    """Load .env file from cwd if it exists. No-op if missing."""
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if value and key not in os.environ:
            os.environ[key] = value


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4"))


def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


class Engine:
    """Executes a Kata execution plan by calling LLM APIs."""

    def __init__(self, verbose: bool = False) -> None:
        self._openai_client = None
        self._anthropic_client = None
        self._verbose = verbose

    # -- lazy client init (only created when needed) -----------------------

    def _get_openai(self):
        if self._openai_client is None:
            import openai
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._openai_client = openai.OpenAI(api_key=key)
        return self._openai_client

    def _get_anthropic(self):
        if self._anthropic_client is None:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            self._anthropic_client = anthropic.Anthropic(api_key=key)
        return self._anthropic_client

    # -- execution ---------------------------------------------------------

    def run(self, plan: ExecutionPlan) -> RunArtifact:
        """Execute all steps in dependency order, return a full artifact."""
        outputs: dict[str, str] = {}

        for step in plan.steps:
            # Chain dependencies are prepended as context;
            # ref dependencies are resolved inline via {{ref:...}} markers
            context = [
                f"[Output from {dep}]\n{outputs[dep]}"
                for dep in step.depends_on
                if dep in outputs and not dep.startswith("ref_")
            ]

            if self._verbose:
                tag = step.model or "default"
                deps = f" <- {', '.join(step.depends_on)}" if step.depends_on else ""
                print(f">> {step.id} ({tag}){deps}", file=sys.stderr)

            outputs[step.id] = self._execute_step(step, context, outputs)

        return RunArtifact(plan=plan, outputs=outputs)

    def _execute_step(
        self, step: Step, context: list[str], all_outputs: dict[str, str] | None = None,
    ) -> str:
        """Build messages and dispatch to the right provider."""
        # System: role/context + constraints
        system_parts: list[str] = []
        if step.system:
            system_parts.append(step.system)
        if step.constraints:
            system_parts.append(
                "Constraints:\n" + "\n".join(f"- {c}" for c in step.constraints)
            )
        system = "\n\n".join(system_parts) if system_parts else None

        # User: prior outputs + task + output format
        user_parts: list[str] = []
        if context:
            user_parts.extend(context)
        if step.user:
            user_text = step.user
            # Resolve inline {{ref:step_id}} markers with actual outputs
            if all_outputs and "{{ref:" in user_text:
                for ref_id, ref_output in all_outputs.items():
                    user_text = user_text.replace(f"{{{{ref:{ref_id}}}}}", ref_output)
            user_parts.append(user_text)
        if step.output:
            fmt = ", ".join(f"{k}: {v}" for k, v in step.output.items())
            user_parts.append(f"Output format: {fmt}")
        user = "\n\n".join(user_parts)

        model = step.model or "gpt-4o"

        if _is_anthropic_model(model):
            return self._call_anthropic(model, system, user)
        if _is_openai_model(model):
            return self._call_openai(model, system, user)

        raise RuntimeError(
            f"Unknown model provider for '{model}'. "
            "Expected 'gpt-*' (OpenAI) or 'claude-*' (Anthropic)."
        )

    # -- provider calls ----------------------------------------------------

    def _call_openai(self, model: str, system: str | None, user: str) -> str:
        client = self._get_openai()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    def _call_anthropic(self, model: str, system: str | None, user: str) -> str:
        client = self._get_anthropic()
        kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text
