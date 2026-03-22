"""Tests for the Kata compiler."""

import unittest
from .lexer import Lexer
from .parser import Parser
from .compiler import Compiler, ExecutionPlan


def compile_source(source: str) -> ExecutionPlan:
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return Compiler().compile(ast)


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


if __name__ == "__main__":
    unittest.main()
