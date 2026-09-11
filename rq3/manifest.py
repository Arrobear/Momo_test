"""Shared manifest for `python -m rq3.run` and diff_seed_generator.

A manifest has a `runs` list. Each run requires run_id, library (import name),
repo_dir (folder under --repository-root), reference and candidate (Git refs).
Optional api_include lists fully qualified API names; input_seeds maps API names
to parameter names to lists of JSON values. generated_input_seeds is an artifact
path relative to the manifest. diff_seed_generation.targets maps API names to
parameter-name lists, with optional paths and max_values_per_parameter fields.
"""

import os
import re
from pathlib import Path

from rq3.json_values import load_json


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def default_root():
    return Path(os.environ.get("MOMO_ROOT_PATH", SOURCE_ROOT.parent)).resolve()


def default_repository_root():
    return Path(os.environ.get("MOMO_REPOSITORY_ROOT", default_root() / "dl_lib")).resolve()


def validate_api_names(names):
    if not isinstance(names, list) or not names or any(
        not isinstance(name, str) or not name or name != name.strip() for name in names
    ):
        raise ValueError("api_include must be a non-empty list of API names")
    if len(set(names)) != len(names):
        raise ValueError("api_include contains duplicate API names")
    return names


def load_run_spec(manifest_path, run_id):
    path = Path(manifest_path).resolve()
    data = load_json(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError("Manifest must contain a runs list")
    if not all(isinstance(run, dict) for run in data["runs"]):
        raise ValueError("Every manifest run must be an object")
    matches = [run for run in data["runs"] if run.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one run named {run_id!r}")
    run = matches[0]
    for name in ("run_id", "library", "repo_dir", "reference", "candidate"):
        value = run.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Run requires a non-empty string: {name}")
    for name in ("run_id", "repo_dir"):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", run[name]):
            raise ValueError(f"{name} must be a single folder name")
    import_name = run.get("import_name", run["library"])
    if not isinstance(import_name, str) or not all(p.isidentifier() for p in import_name.split(".")):
        raise ValueError("library/import_name must be a dotted Python import name")
    if "api_include" in run:
        validate_api_names(run["api_include"])
    return run, path.parent
