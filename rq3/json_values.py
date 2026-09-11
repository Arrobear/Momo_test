"""Strict JSON identities for input deduplication, including nested types."""

import json


def json_key(value):
    # Keep false/0/0.0 distinct and preserve insertion order for Python dict inputs.
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


def load_json(text):
    return json.loads(text, parse_constant=reject_constant)
