"""Built-in role presets for the Kata language.

Usage in .kata files:

    @role :coder            →  "You are a senior software developer."
    @role :coder(Python)    →  "You are a senior Python developer."
    @role :tester           →  "You are a test engineer who writes thorough, readable tests."
    @role :reviewer         →  "You are a thorough code reviewer focused on correctness and security."
    @role :debugger         →  "You are a debugger. You think step by step and narrow down root causes."
    @role :architect        →  "You are a senior software architect."
    @role :architect(API)   →  "You are a senior API architect."
    @role :summarizer       →  "You are a research assistant skilled at distilling complex material."
    @role :writer           →  "You are a technical writer who produces clear, well-structured documentation."
    @role :migrator         →  "You are an expert at migrating codebases between frameworks and libraries."

Any value not starting with ':' is left as-is.
"""

from __future__ import annotations
import re

# template, default fill for {}
_PRESETS: dict[str, tuple[str, str]] = {
    "coder":      ("You are a senior {} developer.", "software"),
    "tester":     ("You are a test engineer who writes thorough, readable tests.", ""),
    "reviewer":   ("You are a thorough code reviewer focused on correctness and security.", ""),
    "debugger":   ("You are a debugger. You think step by step and narrow down root causes.", ""),
    "architect":  ("You are a senior {} architect.", "software"),
    "summarizer": ("You are a research assistant skilled at distilling complex material.", ""),
    "writer":     ("You are a technical writer who produces clear, well-structured documentation.", ""),
    "migrator":   ("You are an expert at migrating codebases between frameworks and libraries.", ""),
}

_PRESET_RE = re.compile(r"^:(\w+)(?:\(([^)]*)\))?\s*$")


def expand_role(value: str) -> str:
    """Expand a role preset like ':coder(Python)' into its full text.

    Returns the value unchanged if it doesn't match the ':preset' syntax.
    Raises ValueError for unknown preset names.
    """
    m = _PRESET_RE.match(value.strip())
    if not m:
        return value

    name, arg = m.group(1), m.group(2)
    entry = _PRESETS.get(name)
    if entry is None:
        known = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown role preset ':{name}'. Known presets: {known}")

    template, default = entry
    if "{}" in template:
        fill = arg.strip() if arg else default
        return template.format(fill)
    return template


def list_presets() -> dict[str, str]:
    """Return a dict of preset name → expanded text (with defaults)."""
    result = {}
    for name, (template, default) in _PRESETS.items():
        if "{}" in template:
            result[name] = template.format(default)
        else:
            result[name] = template
    return result
