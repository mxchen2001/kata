# Kata 型

Structured pseudocode that compiles to LLM API calls.

`.kata` files are prompt guardrails — roles, constraints, and output expectations in a format you can lint, test, and version control. They compile to execution plans, run against OpenAI or Anthropic, and decompile back to source. The output is whatever you asked the model for.

## Why

Most engineers today don't read assembly — they write in high-level languages and let the compiler handle the rest. Kata is a first step toward the same thing happening with code itself: future engineers describe *what* they want, not *how* to implement it, and never read the generated code at all.

## Example

```kata
@model claude-sonnet-4-6
@role :coder(Python)
@task Write a function that checks if a number is prime.
@constraint Use type hints.
@output format: python, file: prime.py
```

## Directives

| Directive | Purpose |
|---|---|
| `@model` | Target model (e.g. `claude-sonnet-4-6`, `gpt-4o`) |
| `@role` | System prompt — supports presets like `:coder`, `:tester`, `:reviewer` |
| `@context` | Additional context block |
| `@task` | The user prompt |
| `@constraint` | Rules the model must follow |
| `@output` | Output format/filename |
| `@fn` / `@call` | Reusable parameterized functions |

## Usage

```sh
kata compile example.kata        # show execution plan
kata check example.kata          # validate syntax
kata run example.kata             # execute against LLM APIs
kata decompile < plan.json       # reverse a plan back to .kata source
```

## Install

```sh
pip install -e .
```

Requires `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in your environment.

## License

ISC
