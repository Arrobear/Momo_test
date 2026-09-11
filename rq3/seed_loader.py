import hashlib
from pathlib import Path

from rq3.json_values import json_key, load_json
from rq3.manifest import validate_api_names


ARTIFACT_SCHEMA = "momo.diff-seeds.v1"


def validate_seed_mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")

    normalized = {}
    for api_name, parameters in value.items():
        if not isinstance(api_name, str) or not api_name:
            raise ValueError(f"{label} contains an invalid API name")
        if not isinstance(parameters, dict):
            raise ValueError(f"{label}.{api_name} must be a mapping")
        normalized[api_name] = {}
        for parameter, seeds in parameters.items():
            if not isinstance(parameter, str) or not parameter:
                raise ValueError(f"{label}.{api_name} contains an invalid parameter")
            if not isinstance(seeds, list):
                raise ValueError(f"{label}.{api_name}.{parameter} must be a list")
            for seed in seeds:
                json_key(seed)
            normalized[api_name][parameter] = list(seeds)
    return normalized


def _merge_seed_mappings(destination, source):
    for api_name, parameters in source.items():
        api_destination = destination.setdefault(api_name, {})
        for parameter, values in parameters.items():
            parameter_destination = api_destination.setdefault(parameter, [])
            known = {json_key(value) for value in parameter_destination}
            for value in values:
                key = json_key(value)
                if key not in known:
                    parameter_destination.append(value)
                    known.add(key)


def _artifact_path(path_value, base_dir):
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def _validate_artifact_scope(artifact, run_spec, path):
    if not isinstance(artifact, dict) or artifact.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"Unsupported diff-seed artifact schema in {path}")

    scope = artifact.get("scope")
    if not isinstance(scope, dict):
        raise ValueError(f"Missing diff-seed scope in {path}")

    expected = {
        "run_id": run_spec.get("run_id"),
        "library": run_spec.get("library"),
        "reference": run_spec.get("reference"),
        "candidate": run_spec.get("candidate"),
        "resolved_reference": run_spec.get("resolved_reference"),
        "resolved_candidate": run_spec.get("resolved_candidate"),
    }
    for key, expected_value in expected.items():
        if expected_value is not None and scope.get(key) != expected_value:
            raise ValueError(
                f"Diff-seed artifact {key} mismatch: "
                f"expected {expected_value!r}, found {scope.get(key)!r}"
            )


def resolve_input_seed_bundle(run_spec, base_dir):
    """Resolve manual and generated seeds without mutating the run specification."""
    merged = {}
    sources = []

    manual = validate_seed_mapping(run_spec.get("input_seeds", {}), "input_seeds")
    if manual:
        _merge_seed_mappings(merged, manual)
        sources.append({"kind": "manifest", "field": "input_seeds"})

    artifact_value = run_spec.get("generated_input_seeds")
    if artifact_value:
        path = _artifact_path(artifact_value, base_dir)
        raw = path.read_bytes()
        artifact = load_json(raw.decode("utf-8"))
        _validate_artifact_scope(artifact, run_spec, path)
        generated = validate_seed_mapping(
            artifact.get("input_seeds", {}), "generated input_seeds"
        )

        artifact_targets = artifact["scope"].get("targets")
        if not isinstance(artifact_targets, dict):
            raise ValueError(f"Diff-seed artifact has no target parameter map: {path}")
        for api_name, parameters in generated.items():
            allowed_parameters = artifact_targets.get(api_name)
            if not isinstance(allowed_parameters, list):
                raise ValueError(f"Generated API is absent from artifact targets: {api_name}")
            unexpected_parameters = sorted(set(parameters) - set(allowed_parameters))
            if unexpected_parameters:
                raise ValueError(
                    f"Generated parameters are outside artifact targets for {api_name}: "
                    + ", ".join(unexpected_parameters)
                )

        allowed_apis = set(validate_api_names(run_spec.get("api_include", [])))
        unexpected = sorted(set(generated) - allowed_apis)
        if unexpected:
            raise ValueError(
                "Generated seeds target APIs outside api_include: "
                + ", ".join(unexpected)
            )

        _merge_seed_mappings(merged, generated)
        sources.append(
            {
                "kind": "diff_seed_artifact",
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "diff_sha256": artifact.get("diff", {}).get("sha256"),
                "information_policy": artifact.get("information_policy"),
            }
        )

    if "api_include" in run_spec:
        unexpected = set(merged) - set(validate_api_names(run_spec["api_include"]))
        if unexpected:
            raise ValueError(f"Input seeds target APIs outside api_include: {sorted(unexpected)}")
    return {"input_seeds": merged, "sources": sources}
