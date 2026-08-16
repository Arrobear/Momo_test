"""LLM feedback loop for path-specific V1 test cases."""

import argparse
import ast
import json
from pathlib import Path

from config import MODEL, make_client
from stage_1_function import call_llm_with_retry, extract_clean_json


def load_json(path):
    with open(str(path), "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(str(temporary), "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def code_is_runnable(code):
    if not isinstance(code, str) or not code.strip():
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    has_entrypoint = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_test_case"
        for node in tree.body
    )
    has_target_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "momo_call"
        for node in ast.walk(tree)
    )
    return has_entrypoint and has_target_call


def automatic_issues(case_data, observation):
    issues = []
    expected = case_data.get("expected_status")
    status = observation.get("函数运行状态")
    phase = observation.get("execution_phase")
    if not observation.get("target_invoked"):
        issues.append("target API was never invoked through momo_call")
    elif not observation.get("target_matches_api"):
        issues.append("momo_call invoked a callable other than the requested API")
    if observation.get("target_call_count") != 1:
        issues.append("test case must invoke exactly one target API call")
    if status in {"harness_error", "eval_failed", "timeout", "recursion_bug"}:
        issues.append("execution status %s is not a valid path oracle" % status)
    if expected == "success":
        if status != "success":
            issues.append("return path expected success but observed %s" % status)
        if not observation.get("target_completed"):
            issues.append("target call did not complete")
    elif expected == "error":
        if status != "error" or phase != "target_error":
            issues.append(
                "raise path requires an exception from the target call; "
                "observed status=%s phase=%s" % (status, phase)
            )
    else:
        issues.append("unknown expected status: %s" % expected)
    return issues


def build_review_prompt(case_data, observation, issues):
    return """
You are validating and repairing one path-specific Python differential test.

API name:
{api_name}

API signature:
{api_signature}

Target Python version:
{required_python}

API documentation:
{api_documentation}

API source:
{api_source}

Parameter conditions:
{parameter_conditions}

Boundary context:
{boundary_context}

Target path id:
{path_id}

Target path constraints:
{path_constraints}

Expected status:
{expected_status}

Current test code:
```python
{code}
```

Observed execution:
{observation}

Deterministic validation issues:
{issues}

Decide whether this code genuinely and reproducibly executes the requested
target path. An expected error is valid only when the target API itself raises
from the requested path. Syntax/import/setup/missing-file/instantiation errors,
timeouts, failures after the target call, and calls that never reach the target
are invalid.

If invalid, provide a complete replacement `run_test_case()` implementation.
It must create all files, objects, instances, loops, and executors it needs and
must invoke the real API through:

    return momo_call(target_callable, *args, **kwargs)

Do not mock the target implementation, catch the target exception, or use an
assertion as the oracle.
The replacement must be compatible with Python {required_python}.

Return only JSON:
{{
  "valid": true or false,
  "reason": "concise technical reason",
  "repaired_code": "complete replacement code when invalid, otherwise null",
  "summary": "short concrete-input summary"
}}
""".format(
        api_name=case_data.get("api_name"),
        api_signature=case_data.get("api_signature"),
        required_python=case_data.get("required_python"),
        api_documentation=case_data.get("api_documentation"),
        api_source=case_data.get("api_source"),
        parameter_conditions=case_data.get("parameter_conditions"),
        boundary_context=case_data.get("boundary_context"),
        path_id=case_data.get("path_id"),
        path_constraints=case_data.get("path_constraints"),
        expected_status=case_data.get("expected_status"),
        code=case_data.get("code"),
        observation=json.dumps(observation, ensure_ascii=False, indent=2),
        issues=json.dumps(issues, ensure_ascii=False),
    )


def parse_review(text):
    payload = extract_clean_json(text) if text else None
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "reason": "LLM review did not return a JSON object",
            "repaired_code": None,
            "summary": "",
        }
    return {
        "valid": payload.get("valid") is True,
        "reason": str(payload.get("reason", "")).strip(),
        "repaired_code": payload.get("repaired_code"),
        "summary": str(payload.get("summary", "")).strip(),
    }


def refine_cases(bundle_dir, attempts_path, round_number, client=None):
    bundle_dir = Path(bundle_dir)
    cases_path = bundle_dir / "test_cases.json"
    cases = load_json(cases_path)
    attempts = load_json(attempts_path)
    client = client or make_client()
    observations = {
        entry.get("case_id"): entry
        for api_attempts in attempts.values()
        for entry in api_attempts
        if isinstance(entry, dict)
    }

    repaired = 0
    for api_name, api_cases in cases.items():
        if not isinstance(api_cases, list):
            continue
        for case_data in api_cases:
            if not isinstance(case_data, dict) or case_data.get("validated"):
                continue
            case_id = case_data.get("case_id")
            observation = observations.get(case_id)
            if observation is None:
                raise RuntimeError("missing V1 observation for %s" % case_id)

            issues = automatic_issues(case_data, observation)
            prompt = build_review_prompt(case_data, observation, issues)
            response = call_llm_with_retry(
                client,
                MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Validate path coverage and repair Python tests. "
                            "Output JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                seed=42 + round_number,
            )
            review = parse_review(response)
            accepted = review["valid"] and not issues
            case_data.setdefault("validation_history", []).append(
                {
                    "round": round_number,
                    "status": observation.get("函数运行状态"),
                    "execution_phase": observation.get("execution_phase"),
                    "target_invoked": observation.get("target_invoked"),
                    "target_call_count": observation.get("target_call_count"),
                    "target_matches_api": observation.get(
                        "target_matches_api"
                    ),
                    "automatic_issues": issues,
                    "model_valid": review["valid"],
                    "model_reason": review["reason"],
                }
            )

            if accepted:
                case_data["validated"] = True
                case_data["validation_reason"] = review["reason"]
                case_data["baseline_observation"] = observation
                if review["summary"]:
                    case_data["summary"] = review["summary"]
                continue

            repaired_code = review.get("repaired_code")
            if code_is_runnable(repaired_code):
                case_data["code"] = repaired_code.strip()
                case_data["revision"] = int(case_data.get("revision", 0)) + 1
                if review["summary"]:
                    case_data["summary"] = review["summary"]
                repaired += 1

    save_json(cases_path, cases)
    all_cases = [
        case_data
        for api_cases in cases.values()
        if isinstance(api_cases, list)
        for case_data in api_cases
        if isinstance(case_data, dict)
        and case_data.get("schema_version") == 2
    ]
    validated = sum(1 for case_data in all_cases if case_data.get("validated"))
    return {
        "round": round_number,
        "total": len(all_cases),
        "validated": validated,
        "pending": len(all_cases) - validated,
        "repaired": repaired,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args(argv)

    status = refine_cases(
        bundle_dir=args.bundle,
        attempts_path=args.attempts,
        round_number=args.round,
    )
    save_json(args.status, status)
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
