"""Keep seed data and its decoding contract together through Momo's JSON/repr IO.

Each injected candidate is {SEED_TAG: encoded_json}. The reserved tag is decoded
before Momo's code/literal heuristics in both V1 and V2. A single tagged value
survives the existing list merge without a parallel type array or baseline change.
"""

import ast
import os
from pathlib import Path

from rq3.json_values import json_key, load_json
from rq3.seed_loader import validate_seed_mapping


SEED_TAG = "__momo_json_seed_v1__"


def configured_input_seeds():
    # Files avoid Windows environment-variable size limits for large seed bundles.
    path = os.environ.get("MOMO_INPUT_SEEDS_FILE")
    raw = Path(path).read_text(encoding="utf-8") if path else os.environ.get("MOMO_INPUT_SEEDS")
    return validate_seed_mapping(load_json(raw), "input_seeds") if raw else {}


def decode_input_seed(value):
    if isinstance(value, str):
        if not value.startswith(("{'" + SEED_TAG + "':", '{"' + SEED_TAG + '":')):
            return False, value
        value = ast.literal_eval(value)
    if isinstance(value, dict) and set(value) == {SEED_TAG}:
        return True, load_json(value[SEED_TAG])
    return False, value


def merge_input_seeds(api_name, candidates, allowed_parameters):
    """Append typed JSON seeds, leaving ordinary Momo candidates untouched."""
    api_seeds = configured_input_seeds().get(api_name, {})
    unexpected = sorted(set(api_seeds) - set(allowed_parameters))
    if unexpected:
        raise ValueError(f"Input seeds target unknown parameters for {api_name}: {unexpected}")
    for parameter, values in api_seeds.items():
        entry = candidates.setdefault(parameter, {"type": "literal", "values": []})
        destination = entry["values"]
        for value in values:
            tagged = {SEED_TAG: json_key(value)}
            if tagged not in destination:
                destination.append(tagged)
    return candidates
