"""Tests for the Kata compiler."""

import os
import tempfile
import unittest
from pathlib import Path
from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler, ExecutionPlan


def compile_source(source: str, base_path: Path | None = None) -> ExecutionPlan:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return Compiler(base_path=base_path).compile(ast)


class TestCompiler(unittest.TestCase):
    def test_simple_program(self):
        source = """
@model gpt-4o
@role You are a helpful assistant.
@task Explain what a monad is.
"""
        plan = compile_source(source)
        self.assertEqual(len(plan.steps), 1)
        s = plan.steps[0]
        self.assertEqual(s.model, "gpt-4o")
        self.assertEqual(s.system, "You are a helpful assistant.")
        self.assertEqual(s.user, "Explain what a monad is.")

    def test_context_and_constraints(self):
        source = """
@role Engineer.
@context {
  TypeScript codebase.
  Uses React.
}
@task Refactor the component.
@constraint Keep it under 50 lines.
"""
        plan = compile_source(source)
        s = plan.steps[0]
        self.assertIn("TypeScript codebase.", s.system or "")
        self.assertIn("Refactor the component.", s.user)
        self.assertIn("Keep it under 50 lines.", s.constraints)

    def test_output_properties(self):
        source = """
@task Write a poem.
@output format: markdown, lang: en
"""
        plan = compile_source(source)
        self.assertEqual(plan.steps[0].output["format"], "markdown")
        self.assertEqual(plan.steps[0].output["lang"], "en")

    def test_multi_model_functions(self):
        source = """
@fn summarize {
  @model gpt-4o
  @task Summarize the input.
}

@fn translate {
  @model claude-sonnet-4-6
  @task Translate to Spanish.
}

@call summarize
@call translate
"""
        plan = compile_source(source)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].model, "gpt-4o")
        self.assertEqual(plan.steps[1].model, "claude-sonnet-4-6")
        # step_2 depends on step_1
        self.assertEqual(plan.steps[1].depends_on, ["step_1"])

    def test_function_params(self):
        source = """
@model gpt-4o
@fn review(lang, focus) {
  @task Review ${lang} code for ${focus}.
}
@call review(TypeScript, security)
"""
        plan = compile_source(source)
        self.assertIn("Review TypeScript code for security.", plan.steps[0].user)

    def test_plan_serialization(self):
        source = """
@model gpt-4o
@role Helper.
@task Do something.
@constraint Be concise.
@output format: json
"""
        plan = compile_source(source)
        d = plan.to_dict()
        self.assertIn("plan", d)
        step = d["plan"][0]
        self.assertEqual(step["id"], "step_1")
        self.assertEqual(step["model"], "gpt-4o")
        self.assertEqual(step["constraints"], ["Be concise."])
        self.assertEqual(step["output"], {"format": "json"})


class TestOptimization(unittest.TestCase):
    def test_removes_empty_model_step(self):
        """Top-level @model is redundant when all functions have their own @model."""
        source = """
@fn review(lang, focus) {
  @model claude-sonnet-4-6
  @task Review ${lang} code for ${focus}.
}

@fn explain(audience) {
  @model gpt-4o
  @task Explain to ${audience}.
}

@model claude-sonnet-4-6
@call review(TypeScript, security)
@call explain(developers)
"""
        plan = compile_source(source)
        # The bare @model step should be optimized away
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].id, "step_1")
        self.assertEqual(plan.steps[0].model, "claude-sonnet-4-6")
        self.assertIn("TypeScript", plan.steps[0].user)
        self.assertEqual(plan.steps[1].id, "step_2")
        self.assertEqual(plan.steps[1].model, "gpt-4o")
        # step_2 depends on step_1 (rewired past removed step)
        self.assertEqual(plan.steps[1].depends_on, ["step_1"])

    def test_no_removal_when_step_has_content(self):
        """Steps with content should never be removed."""
        source = """
@model gpt-4o
@role You are a helpful assistant.
@task Do something.
"""
        plan = compile_source(source)
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].user, "Do something.")


class TestContextFile(unittest.TestCase):
    def test_context_file_single(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "data.txt").write_text("hello world")
            source = """
@model gpt-4o
@context file: data.txt
@task Summarize.
"""
            plan = compile_source(source, base_path=Path(tmpdir))
            self.assertIn("hello world", plan.steps[0].system or "")
            self.assertIn("[data.txt]", plan.steps[0].system or "")

    def test_context_file_glob(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("def a(): pass")
            (Path(tmpdir) / "b.py").write_text("def b(): pass")
            (Path(tmpdir) / "c.txt").write_text("ignore me")
            source = """
@model gpt-4o
@context file: *.py
@task Review.
"""
            plan = compile_source(source, base_path=Path(tmpdir))
            system = plan.steps[0].system or ""
            self.assertIn("def a(): pass", system)
            self.assertIn("def b(): pass", system)
            self.assertNotIn("ignore me", system)

    def test_context_file_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = """
@model gpt-4o
@context file: missing.txt
@task Summarize.
"""
            plan = compile_source(source, base_path=Path(tmpdir))
            self.assertIn("file not found", plan.steps[0].system or "")


class TestRetry(unittest.TestCase):
    def test_retry_directive(self):
        source = """
@model gpt-4o
@task Do something.
@retry 5
"""
        plan = compile_source(source)
        self.assertEqual(plan.steps[0].retries, 5)

    def test_retry_in_serialization(self):
        source = """
@model gpt-4o
@task Do something.
@retry 3
"""
        plan = compile_source(source)
        d = plan.to_dict()
        self.assertEqual(d["plan"][0]["retries"], 3)

    def test_no_retry_default(self):
        source = """
@model gpt-4o
@task Do something.
"""
        plan = compile_source(source)
        self.assertEqual(plan.steps[0].retries, 0)
        self.assertNotIn("retries", plan.to_dict()["plan"][0])


class TestImport(unittest.TestCase):
    def test_import_fn_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = Path(tmpdir) / "lib.kata"
            lib.write_text("""
@fn greet(name) {
  @task Say hello to ${name}.
}
""")
            source = """
@import lib
@model gpt-4o
@call greet(World)
"""
            plan = compile_source(source, base_path=Path(tmpdir))
            self.assertEqual(len(plan.steps), 1)
            self.assertIn("Say hello to World.", plan.steps[0].user)

    def test_import_with_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = Path(tmpdir) / "helpers.kata"
            lib.write_text("""
@fn add_role {
  @role You are a helpful assistant.
}
""")
            source = """
@import helpers.kata
@model gpt-4o
@call add_role
@task Do something.
"""
            plan = compile_source(source, base_path=Path(tmpdir))
            self.assertEqual(plan.steps[0].system, "You are a helpful assistant.")

    def test_local_fn_overrides_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib = Path(tmpdir) / "lib.kata"
            lib.write_text("""
@fn greet {
  @task Imported hello.
}
""")
            source = """
@import lib
@fn greet {
  @task Local hello.
}
@model gpt-4o
@call greet
"""
            plan = compile_source(source, base_path=Path(tmpdir))
            self.assertIn("Local hello.", plan.steps[0].user)


if __name__ == "__main__":
    unittest.main()
