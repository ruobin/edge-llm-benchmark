"""Small, deliberately strict checks for the benchmark's fixed-answer tasks."""

import json


def evaluate(text, rule):
    """Return pass/fail/unscored; never execute or judge generated code."""
    if rule is None:
        return {"status": "unscored", "reason": "requires human review"}
    kind = rule["type"]
    if kind == "exact":
        actual = " ".join(text.strip().split()).casefold()
        expected = " ".join(rule["value"].strip().split()).casefold()
        passed = actual == expected
    elif kind == "json":
        try:
            actual = json.loads(text)
        except json.JSONDecodeError:
            return {"status": "fail", "reason": "invalid JSON"}
        passed = actual == rule["value"]
    else:
        raise ValueError(f"Unsupported scoring rule: {kind}")
    return {"status": "pass" if passed else "fail", "reason": "strict deterministic check"}
