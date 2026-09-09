"""LLM feedback loop for path-specific V1 test cases."""

import argparse
import ast
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import MODEL, make_client
from stage_1_function import (
    call_llm_with_retry,
    extract_clean_json,
    get_llm_worker_count,
)


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


def is_target_timeout_observation(observation):
    trace = observation.get("target_trace") or {}
    return (
        observation.get("函数运行状态") == "timeout"
        and observation.get("execution_phase") == "target_timeout"
        and observation.get("target_invoked")
        and observation.get("target_matches_api")
        and observation.get("target_call_count") == 1
        and bool(trace.get("executed_line_count"))
    )


def automatic_issues(case_data, observation):
    issues = []
    expected = case_data.get("expected_status")
    status = observation.get("函数运行状态")
    phase = observation.get("execution_phase")
    trace = observation.get("target_trace") or {}
    bug_context = case_data.get("bug_context") or {}
    bug_patch = str(bug_context.get("bug_patch") or "").strip()
    bug_triggering_target_error = (
        status == "error"
        and phase == "target_error"
        and observation.get("target_invoked")
        and observation.get("target_matches_api")
        and trace.get("executed_line_count")
        and bool(bug_patch)
    )
    target_timeout = is_target_timeout_observation(observation)
    if not observation.get("target_invoked"):
        issues.append("target API was never invoked through momo_call")
    elif not observation.get("target_matches_api"):
        issues.append("momo_call invoked a callable other than the requested API")
    if observation.get("target_call_count") != 1:
        issues.append("test case must invoke exactly one target API call")
    if not trace.get("executed_line_count"):
        issues.append("target source body was not executed under trace")
    if target_timeout:
        if expected not in {"success", "error"}:
            issues.append("unknown expected status: %s" % expected)
        return issues
    if status in {"harness_error", "eval_failed", "timeout", "recursion_bug"}:
        issues.append("execution status %s is not a valid path oracle" % status)
    if expected == "success":
        if status != "success" and not bug_triggering_target_error:
            issues.append("return path expected success but observed %s" % status)
        if not observation.get("target_completed") and not bug_triggering_target_error:
            issues.append("target call did not complete")
        if status == "success" and observation.get("函数返回结果") is None:
            issues.append(
                "success path returned None; return a structured oracle "
                "with observable side effects such as file contents, cache "
                "state, report events, or injected exception status"
            )
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

BugsInPy bug context:
{bug_context}

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
target path. Do not judge validity from success status alone. Use
`target_trace.executed_source` and the requested `path_constraints` to decide
whether the target function body actually executed the intended branch/path.
If the trace is empty or the executed lines do not support the requested path,
the case is invalid even if the return status is "success".

For success paths, returning None is not a useful differential oracle. The test
must return a structured, JSON-serializable oracle derived from observable
behavior after the target call: file contents, cache state, report events,
exception class/message from an injected failure, or other side effects relevant
to the path. An expected error is valid only when the target API itself raises
from the requested path. Syntax/import/setup/missing-file/instantiation errors,
setup timeouts, failures after the target call, and calls that never reach the
target are invalid. A timeout after the requested target API was invoked and
traced is a terminal V1 observation and should be recorded as a timeout bug
candidate, not repaired.

Use the BugsInPy patch context and original run_test command when present.
A case that covers only a generic path but does not exercise or observe the
patched behavior is incomplete for regression detection.

If invalid, provide a complete replacement `run_test_case()` implementation.
It must create all files, objects, instances, loops, and executors it needs and
must invoke the real API through:

    return momo_call(target_callable, *args, **kwargs)

For an async target, do not wrap it. Pass the requested API directly to
`momo_call`, then await the returned coroutine with the prepared event loop:

    awaitable = momo_call(target_async_callable, *args, **kwargs)
    return loop.run_until_complete(awaitable)

Never pass a wrapper around the requested API to `momo_call`.

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
        bug_context=case_data.get("bug_context"),
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


_thread_local_llm = threading.local()


def _thread_llm_client():
    client = getattr(_thread_local_llm, "client", None)
    if client is None:
        client = make_client()
        _thread_local_llm.client = client
    return client


def _review_case_with_llm(task, client=None):
    prompt = build_review_prompt(
        task["case_data"],
        task["observation"],
        task["issues"],
    )
    response = call_llm_with_retry(
        client or _thread_llm_client(),
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
        seed=42 + task["round_number"],
    )
    return task, parse_review(response)


def refine_cases(bundle_dir, attempts_path, round_number, client=None):
    bundle_dir = Path(bundle_dir)
    cases_path = bundle_dir / "test_cases.json"
    cases = load_json(cases_path)
    attempts = load_json(attempts_path)
    observations = {
        entry.get("case_id"): entry
        for api_attempts in attempts.values()
        for entry in api_attempts
        if isinstance(entry, dict)
    }

    repaired = 0
    review_tasks = []
    for api_order, (api_name, api_cases) in enumerate(cases.items()):
        if not isinstance(api_cases, list):
            continue
        for case_index, case_data in enumerate(api_cases):
            if not isinstance(case_data, dict) or case_data.get("validated"):
                continue
            case_id = case_data.get("case_id")
            observation = observations.get(case_id)
            if observation is None:
                raise RuntimeError("missing V1 observation for %s" % case_id)

            issues = automatic_issues(case_data, observation)
            if is_target_timeout_observation(observation) and not issues:
                case_data.setdefault("validation_history", []).append(
                    {
                        "round": round_number,
                        "status": observation.get("函数运行状态"),
                        "execution_phase": observation.get("execution_phase"),
                        "target_invoked": observation.get("target_invoked"),
                        "target_call_count": observation.get(
                            "target_call_count"
                        ),
                        "target_matches_api": observation.get(
                            "target_matches_api"
                        ),
                        "target_trace_line_count": (
                            observation.get("target_trace") or {}
                        ).get("executed_line_count"),
                        "automatic_issues": issues,
                        "model_valid": None,
                        "model_reason": (
                            "accepted automatically as target timeout"
                        ),
                    }
                )
                case_data["validated"] = True
                case_data["validation_reason"] = (
                    "target API timed out during V1 probe"
                )
                case_data["baseline_observation"] = observation
                continue
            review_tasks.append(
                {
                    "api_order": api_order,
                    "api_name": api_name,
                    "case_index": case_index,
                    "case_data": case_data,
                    "observation": observation,
                    "issues": issues,
                    "round_number": round_number,
                }
            )

    llm_workers = get_llm_worker_count()
    review_results = []
    if review_tasks:
        print(
            "Parallel LLM review/repair tasks: "
            f"{len(review_tasks)}, workers={llm_workers}"
        )
        if llm_workers == 1 or len(review_tasks) <= 1:
            serial_client = client or make_client()
            for task in review_tasks:
                review_results.append(_review_case_with_llm(task, serial_client))
                print(
                    "LLM review/repair progress: "
                    f"{len(review_results)}/{len(review_tasks)}"
                )
        else:
            with ThreadPoolExecutor(max_workers=llm_workers) as executor:
                futures = [
                    executor.submit(_review_case_with_llm, task)
                    for task in review_tasks
                ]
                for future in as_completed(futures):
                    review_results.append(future.result())
                    print(
                        "LLM review/repair progress: "
                        f"{len(review_results)}/{len(review_tasks)}"
                    )

    for task, review in sorted(
        review_results,
        key=lambda item: (item[0]["api_order"], item[0]["case_index"]),
    ):
        case_data = task["case_data"]
        observation = task["observation"]
        issues = task["issues"]
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
                "target_trace_line_count": (
                    observation.get("target_trace") or {}
                ).get("executed_line_count"),
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
