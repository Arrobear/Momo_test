"""Minimal differential-test executor for target Python environments.

This module intentionally uses only Python 3.6-compatible syntax and the
standard library. Target-library dependencies are installed in the isolated
environment before this script is launched.
"""

import argparse
import ast
import asyncio
import importlib
import inspect
import json
import os
import platform
import random
import re
import signal
import sys
import traceback
from pathlib import Path


class CaseTimeout(Exception):
    pass


def load_json(path, default=None):
    try:
        with open(str(path), "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError):
        return default


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def exact_python_version():
    return ".".join(str(item) for item in sys.version_info[:3])


def assert_python_version(expected):
    actual = exact_python_version()
    if actual != expected:
        raise RuntimeError(
            "strict Python version mismatch: expected=%s actual=%s executable=%s"
            % (expected, actual, sys.executable)
        )


def runtime_api_name_candidates(api_name):
    candidates = []
    for prefix in ("lib.", "src."):
        if api_name.startswith(prefix):
            candidates.append(api_name[len(prefix):])
    candidates.append(api_name)
    result = []
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
    return result


def resolve_api_callable(api_name):
    for candidate in runtime_api_name_candidates(api_name):
        target = _resolve_api_callable(candidate)
        if target is not None:
            return target
    return None


def _resolve_api_callable(api_name):
    parts = api_name.split(".")
    for module_end in range(len(parts) - 1, 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:module_end]))
        except ImportError:
            continue
        try:
            for attribute in parts[module_end:]:
                target = getattr(target, attribute)
        except AttributeError:
            continue
        if callable(target):
            return target
    return None


def extract_run_api_code(case_text):
    if isinstance(case_text, dict):
        for value in case_text.values():
            result = extract_run_api_code(value)
            if result:
                return result
        return None
    if isinstance(case_text, list):
        for value in case_text:
            result = extract_run_api_code(value)
            if result:
                return result
        return None
    if not isinstance(case_text, str):
        return None
    match = re.search(
        r"```(?:python)?\s*(.*?)```",
        case_text,
        re.DOTALL | re.IGNORECASE,
    )
    candidate = match.group(1).strip() if match else case_text.strip()
    if re.search(r"^\s*def\s+run_api\s*\(", candidate, re.MULTILINE):
        return candidate
    return None


def build_eval_globals(api_name, lib_name):
    namespace = {"__builtins__": __builtins__}
    root_names = [api_name.split(".", 1)[0], lib_name]
    for root_name in root_names:
        try:
            module = importlib.import_module(root_name)
        except ImportError:
            continue
        namespace[root_name] = module
        for name in dir(module):
            if name.startswith("_"):
                continue
            try:
                namespace.setdefault(name, getattr(module, name))
            except Exception:
                pass

    common_modules = {
        "asyncio": "asyncio",
        "concurrent": "concurrent",
        "pathlib": "pathlib",
        "typing": "typing",
        "numpy": "numpy",
        "np": "numpy",
        "torch": "torch",
    }
    for alias, module_name in common_modules.items():
        try:
            namespace[alias] = importlib.import_module(module_name)
        except ImportError:
            pass
    try:
        namespace["Path"] = importlib.import_module("pathlib").Path
    except Exception:
        pass
    return namespace


def load_run_api(api_name, lib_name, case_text):
    target = resolve_api_callable(api_name)
    code = extract_run_api_code(case_text)
    if code:
        namespace = build_eval_globals(api_name, lib_name)
        try:
            exec(code, namespace)
            generated = namespace.get("run_api")
            if callable(generated):
                return generated, target, "generated"
        except Exception as error:
            print("generated run_api failed for %s: %s" % (api_name, error))
    if target is None:
        return None, None, "unresolved"

    def direct_run_api(*args, **kwargs):
        return target(*args, **kwargs)

    return direct_run_api, target, "direct"


def safe_serialize(value):
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    try:
        return repr(value)
    except Exception as error:
        return "[SERIALIZE_ERROR] %s: %s" % (type(error).__name__, error)


def evaluate_value(value, value_type, namespace):
    if not isinstance(value, str):
        return value, True, None
    stripped = value.strip()
    if value_type != "code":
        if stripped in ("None", "null"):
            return None, True, None
        if stripped == "True":
            return True, True, None
        if stripped == "False":
            return False, True, None
        if stripped and stripped[0] in "{[(":
            try:
                return ast.literal_eval(stripped), True, None
            except (ValueError, SyntaxError):
                pass
        if not ("(" in stripped and ")" in stripped):
            return value, True, None
    try:
        result = eval(stripped, namespace)
        return result, True, None
    except Exception as error:
        return value, False, "%s: %s" % (type(error).__name__, error)


def function_parameters(target):
    if target is None:
        return None
    try:
        return set(inspect.signature(target).parameters.keys())
    except (TypeError, ValueError):
        return None


def run_with_timeout(callable_object, timeout, kwargs):
    if not hasattr(signal, "SIGALRM"):
        return callable_object(**kwargs)

    def handler(signum, frame):
        raise CaseTimeout("API execution exceeded %ss" % timeout)

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return callable_object(**kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def assemble_case(inputs_dict, counts):
    assembled = {}
    for parameter_name, parameter_info in inputs_dict.items():
        if isinstance(parameter_info, dict):
            values = parameter_info.get("values", [])
            value_type = parameter_info.get("type", "literal")
        else:
            values = parameter_info
            value_type = "literal"
        if not values:
            continue
        parameter_counts = counts.setdefault(parameter_name, [0] * len(values))
        weights = [1.0 / (count + 1) for count in parameter_counts]
        selected = random.choices(range(len(values)), weights=weights, k=1)[0]
        parameter_counts[selected] += 1
        assembled[parameter_name] = {
            "value": values[selected],
            "type": value_type,
        }
    return assembled


def execute_case(run_api, target, serialized_input, input_types, namespace, timeout):
    entry = {
        "测试输入": serialized_input,
        "函数返回结果": None,
        "函数运行状态": "pending",
    }
    evaluated = {}
    for parameter_name, value in serialized_input.items():
        result, ok, error = evaluate_value(
            value,
            input_types.get(parameter_name, "literal"),
            namespace,
        )
        if not ok:
            entry["函数运行状态"] = "eval_failed"
            entry["函数返回结果"] = (
                "[EVAL_FAILED] parameter %s: %s" % (parameter_name, error)
            )
            return entry
        evaluated[parameter_name] = result

    valid_parameters = function_parameters(target)
    if valid_parameters is not None:
        evaluated = {
            name: value
            for name, value in evaluated.items()
            if name in valid_parameters
        }

    try:
        output = run_with_timeout(run_api, timeout, evaluated)
        entry["函数运行状态"] = "success"
        entry["函数返回结果"] = safe_serialize(output)
    except CaseTimeout as error:
        entry["函数运行状态"] = "timeout"
        entry["函数返回结果"] = "[TIMEOUT] %s" % error
    except RecursionError as error:
        entry["函数运行状态"] = "recursion_bug"
        entry["函数返回结果"] = "[RECURSION_BUG] %s" % error
    except BaseException as error:
        entry["函数运行状态"] = "error"
        try:
            entry["函数返回结果"] = "%s: %s" % (
                type(error).__name__,
                str(error),
            )
        except Exception:
            entry["函数返回结果"] = "%s: <str failed>" % type(error).__name__
    return entry


def input_types(inputs_dict):
    result = {}
    for name, info in inputs_dict.items():
        result[name] = info.get("type", "literal") if isinstance(info, dict) else "literal"
    return result


def collect_bug_report(baseline, status):
    report = {}
    for api_name, cases in baseline.items():
        selected = []
        for index, case in enumerate(cases):
            if case.get("函数运行状态") == status:
                selected.append(
                    {
                        "case_index": index,
                        "inputs": case.get("测试输入", {}),
                        "detail": case.get("函数返回结果"),
                    }
                )
        if selected:
            report[api_name] = selected
    return report


def has_path_cases(cases, record):
    for api_name in record.get("api_names", []):
        api_cases = cases.get(api_name)
        if (
            isinstance(api_cases, list)
            and api_cases
            and isinstance(api_cases[0], dict)
            and api_cases[0].get("schema_version") == 2
        ):
            return True
    return False


def target_matches_api(target, expected_target):
    if expected_target is None:
        return False
    if target is expected_target:
        return True
    bound_function = getattr(target, "__func__", None)
    if bound_function is expected_target:
        return True
    expected_function = getattr(expected_target, "__func__", None)
    if expected_function is not None and bound_function is expected_function:
        return True
    return False


def target_source_info(target):
    if target is None:
        return None
    function = getattr(target, "__func__", target)
    try:
        source_file = inspect.getsourcefile(function)
        source_lines, start_line = inspect.getsourcelines(function)
    except (OSError, TypeError):
        return None
    if source_file is None:
        return None
    source_map = {}
    for offset, line in enumerate(source_lines):
        line_no = start_line + offset
        source_map[line_no] = line.rstrip("\n")
    return {
        "file": os.path.abspath(source_file),
        "start_line": start_line,
        "end_line": start_line + len(source_lines) - 1,
        "source_map": source_map,
    }


def public_trace(trace_state):
    source_map = trace_state.get("source_map", {})
    executed = sorted(set(trace_state.get("executed_lines", [])))
    executed_source = [
        {
            "line": line_no,
            "code": source_map.get(line_no, ""),
        }
        for line_no in executed
    ]
    return {
        "file": trace_state.get("file"),
        "start_line": trace_state.get("start_line"),
        "end_line": trace_state.get("end_line"),
        "executed_lines": executed,
        "executed_line_count": len(executed),
        "executed_source": executed_source,
    }


def _execute_path_case_inner(api_name, lib_name, case_data, timeout):
    code = case_data.get("code", "")
    entry = {
        "case_id": case_data.get("case_id"),
        "测试输入": {"summary": case_data.get("summary", "")},
        "函数返回结果": None,
        "函数运行状态": "pending",
        "target_invoked": False,
        "target_call_count": 0,
        "target_completed": False,
        "target_matches_api": False,
        "execution_phase": "compile",
        "traceback": "",
        "harness_source": "path_case",
    }
    if not isinstance(code, str) or not code.strip():
        entry["函数运行状态"] = "harness_error"
        entry["函数返回结果"] = "path test case code is empty"
        return entry

    namespace = build_eval_globals(api_name, lib_name)
    expected_target = resolve_api_callable(api_name)
    namespace["target_callable"] = expected_target
    trace_info = target_source_info(expected_target)
    trace_state = {
        "file": trace_info.get("file") if trace_info else None,
        "start_line": trace_info.get("start_line") if trace_info else None,
        "end_line": trace_info.get("end_line") if trace_info else None,
        "source_map": trace_info.get("source_map", {}) if trace_info else {},
        "executed_lines": [],
    }
    call_state = {
        "invoked": False,
        "call_count": 0,
        "completed": False,
        "raised": False,
        "target": None,
        "target_matches_api": False,
    }

    def momo_call(target, *args, **kwargs):
        if not callable(target):
            raise TypeError("momo_call target is not callable")
        call_state["invoked"] = True
        call_state["call_count"] += 1
        call_state["target"] = safe_serialize(target)
        call_state["target_matches_api"] = target_matches_api(
            target, expected_target
        )
        try:
            result = target(*args, **kwargs)
        except BaseException:
            call_state["raised"] = True
            raise
        call_state["completed"] = True
        return result

    namespace["momo_call"] = momo_call
    try:
        compiled = compile(code, "<momo-path-case>", "exec")
        exec(compiled, namespace)
        namespace["momo_call"] = momo_call
    except BaseException as error:
        entry["函数运行状态"] = "harness_error"
        entry["函数返回结果"] = "%s: %s" % (type(error).__name__, error)
        entry["traceback"] = traceback.format_exc()
        return entry

    run_test_case = namespace.get("run_test_case")
    if not callable(run_test_case):
        entry["函数运行状态"] = "harness_error"
        entry["函数返回结果"] = "generated code did not define run_test_case()"
        return entry

    def trace_calls(frame, event, arg):
        if event != "line" or trace_state["file"] is None:
            return trace_calls
        filename = os.path.abspath(frame.f_code.co_filename)
        if filename != trace_state["file"]:
            return trace_calls
        line_no = frame.f_lineno
        if trace_state["start_line"] <= line_no <= trace_state["end_line"]:
            trace_state["executed_lines"].append(line_no)
        return trace_calls

    entry["execution_phase"] = "setup"
    previous_trace = sys.gettrace()
    sys.settrace(trace_calls)
    try:
        output = run_with_timeout(run_test_case, timeout, {})
        entry["target_trace"] = public_trace(trace_state)
        entry["target_invoked"] = call_state["invoked"]
        entry["target_call_count"] = call_state["call_count"]
        entry["target_completed"] = call_state["completed"]
        entry["target_callable"] = call_state["target"]
        entry["target_matches_api"] = call_state["target_matches_api"]
        if not call_state["invoked"]:
            entry["函数运行状态"] = "harness_error"
            entry["函数返回结果"] = "run_test_case returned without momo_call()"
            return entry
        if inspect.isawaitable(output):
            close_awaitable = getattr(output, "close", None)
            if callable(close_awaitable):
                close_awaitable()
            entry["target_completed"] = False
            entry["execution_phase"] = "target_not_awaited"
            entry["函数运行状态"] = "harness_error"
            entry["函数返回结果"] = (
                "target API returned an awaitable that run_test_case "
                "did not execute"
            )
            return entry
        entry["execution_phase"] = "target"
        entry["函数运行状态"] = "success"
        entry["函数返回结果"] = safe_serialize(output)
    except CaseTimeout as error:
        entry["target_trace"] = public_trace(trace_state)
        entry["target_invoked"] = call_state["invoked"]
        entry["target_call_count"] = call_state["call_count"]
        entry["target_completed"] = call_state["completed"]
        entry["target_callable"] = call_state["target"]
        entry["target_matches_api"] = call_state["target_matches_api"]
        entry["execution_phase"] = (
            "target_timeout" if call_state["invoked"] else "setup_timeout"
        )
        entry["函数运行状态"] = "timeout"
        entry["函数返回结果"] = "[TIMEOUT] %s" % error
        entry["traceback"] = traceback.format_exc()
    except RecursionError as error:
        entry["target_trace"] = public_trace(trace_state)
        entry["target_invoked"] = call_state["invoked"]
        entry["target_call_count"] = call_state["call_count"]
        entry["target_completed"] = call_state["completed"]
        entry["target_callable"] = call_state["target"]
        entry["target_matches_api"] = call_state["target_matches_api"]
        entry["execution_phase"] = (
            "target_error" if call_state["raised"] else "harness_error"
        )
        entry["函数运行状态"] = "recursion_bug"
        entry["函数返回结果"] = "[RECURSION_BUG] %s" % error
        entry["traceback"] = traceback.format_exc()
    except BaseException as error:
        entry["target_trace"] = public_trace(trace_state)
        entry["target_invoked"] = call_state["invoked"]
        entry["target_call_count"] = call_state["call_count"]
        entry["target_completed"] = call_state["completed"]
        entry["target_callable"] = call_state["target"]
        entry["target_matches_api"] = call_state["target_matches_api"]
        if call_state["raised"]:
            entry["execution_phase"] = "target_error"
            entry["函数运行状态"] = "error"
        else:
            entry["execution_phase"] = (
                "post_target_error"
                if call_state["completed"]
                else "setup_error"
            )
            entry["函数运行状态"] = "harness_error"
        try:
            entry["函数返回结果"] = "%s: %s" % (
                type(error).__name__,
                str(error),
            )
        except Exception:
            entry["函数返回结果"] = "%s: <str failed>" % type(error).__name__
        entry["traceback"] = traceback.format_exc()
    finally:
        sys.settrace(previous_trace)
        if "target_trace" not in entry:
            entry["target_trace"] = public_trace(trace_state)
    return entry


def execute_path_case(api_name, lib_name, case_data, timeout):
    original_platform = sys.platform
    original_cwd = os.getcwd()
    original_environ = dict(os.environ)
    original_sys_path = list(sys.path)
    try:
        original_loop = asyncio.get_event_loop()
    except RuntimeError:
        original_loop = None

    try:
        return _execute_path_case_inner(
            api_name,
            lib_name,
            case_data,
            timeout,
        )
    finally:
        sys.platform = original_platform
        sys.path[:] = original_sys_path
        os.environ.clear()
        os.environ.update(original_environ)
        try:
            os.chdir(original_cwd)
        except OSError:
            pass
        try:
            current_loop = asyncio.get_event_loop()
        except RuntimeError:
            current_loop = None
        if (
            current_loop is not None
            and current_loop is not original_loop
            and not current_loop.is_running()
            and not current_loop.is_closed()
        ):
            current_loop.close()
        asyncio.set_event_loop(original_loop)


def run_probe(bundle, result_dir, timeout):
    record = load_json(bundle / "record.json", {})
    cases = load_json(bundle / "test_cases.json", {})
    if not has_path_cases(cases, record):
        raise RuntimeError("probe mode requires schema_version=2 path test cases")

    attempts = {}
    for api_name in record.get("api_names", []):
        pending_cases = []
        for case_data in cases.get(api_name, []):
            if not isinstance(case_data, dict) or case_data.get("validated"):
                continue
            pending_cases.append(
                execute_path_case(
                    api_name,
                    record["lib_name"],
                    case_data,
                    timeout,
                )
            )
        if pending_cases:
            attempts[api_name] = pending_cases
    save_json(result_dir / "v1_attempts.json", attempts)


def observation_matches_validated_case(case_data, observation):
    if not observation.get("target_matches_api"):
        return False
    if observation.get("target_call_count") != 1:
        return False
    trace = observation.get("target_trace") or {}
    target_timeout = (
        observation.get("函数运行状态") == "timeout"
        and observation.get("execution_phase") == "target_timeout"
        and observation.get("target_invoked")
        and bool(trace.get("executed_line_count"))
    )
    if target_timeout:
        return True
    bug_context = case_data.get("bug_context") or {}
    bug_triggering_target_error = (
        observation.get("函数运行状态") == "error"
        and observation.get("execution_phase") == "target_error"
        and observation.get("target_invoked")
        and trace.get("executed_line_count")
        and bool(str(bug_context.get("bug_patch") or "").strip())
    )
    expected = case_data.get("expected_status")
    status = observation.get("函数运行状态")
    if expected == "success":
        return (
            status == "success"
            and observation.get("target_completed")
            or bug_triggering_target_error
        )
    if expected == "error":
        return (
            status == "error"
            and observation.get("execution_phase") == "target_error"
        )
    return False


def replay_validated_path_cases(bundle, timeout):
    record = load_json(bundle / "record.json", {})
    cases_path = bundle / "test_cases.json"
    cases = load_json(cases_path, {})
    for api_name in record.get("api_names", []):
        for case_data in cases.get(api_name, []):
            if not case_data.get("validated"):
                raise RuntimeError(
                    "path case was not validated: %s"
                    % case_data.get("case_id")
                )
            observation = execute_path_case(
                api_name,
                record["lib_name"],
                case_data,
                timeout,
            )
            if not observation_matches_validated_case(
                case_data, observation
            ):
                raise RuntimeError(
                    "validated path case was not reproducible: %s "
                    "(status=%s phase=%s)"
                    % (
                        case_data.get("case_id"),
                        observation.get("函数运行状态"),
                        observation.get("execution_phase"),
                    )
                )
            case_data["baseline_observation"] = observation
    save_json(cases_path, cases)


def materialize_path_baseline(bundle, result_dir):
    record = load_json(bundle / "record.json", {})
    cases = load_json(bundle / "test_cases.json", {})
    baseline = {}
    for api_name in record.get("api_names", []):
        baseline_cases = []
        for case_data in cases.get(api_name, []):
            if not case_data.get("validated"):
                raise RuntimeError(
                    "path case was not validated: %s" % case_data.get("case_id")
                )
            observation = case_data.get("baseline_observation")
            if not isinstance(observation, dict):
                raise RuntimeError(
                    "validated path case has no observation: %s"
                    % case_data.get("case_id")
                )
            entry = {
                "case_id": case_data.get("case_id"),
                "path_id": case_data.get("path_id"),
                "path_type": case_data.get("path_type"),
                "path_constraints": case_data.get("path_constraints", []),
                "expected_status": case_data.get("expected_status"),
                "test_case_code": case_data.get("code"),
                "测试用例": case_data.get("code"),
                "validation_reason": case_data.get("validation_reason", ""),
                "validation_history": case_data.get(
                    "validation_history", []
                ),
                "revision": case_data.get("revision", 0),
            }
            entry.update(observation)
            baseline_cases.append(entry)
        baseline[api_name] = baseline_cases

    save_json(result_dir / "v1_baseline.json", baseline)
    save_json(
        result_dir / "recursion_bugs.json",
        collect_bug_report(baseline, "recursion_bug"),
    )
    save_json(
        result_dir / "timeout_bugs.json",
        collect_bug_report(baseline, "timeout"),
    )


def run_v1_legacy(bundle, result_dir, count, timeout):
    record = load_json(bundle / "record.json", {})
    inputs = load_json(bundle / "inputs.json", {})
    cases = load_json(bundle / "test_cases.json", {})
    lib_name = record["lib_name"]
    baseline = {}
    random.seed(42)

    for api_name in record["api_names"]:
        api_inputs = inputs.get(api_name)
        if not isinstance(api_inputs, dict):
            baseline[api_name] = [
                {
                    "测试输入": {},
                    "函数返回结果": "input data missing",
                    "函数运行状态": "harness_error",
                }
            ]
            continue
        run_api, target, source = load_run_api(
            api_name,
            lib_name,
            cases.get(api_name),
        )
        if run_api is None:
            baseline[api_name] = [
                {
                    "测试输入": {},
                    "函数返回结果": "run_api function loading failed",
                    "函数运行状态": "harness_error",
                }
            ]
            continue

        namespace = build_eval_globals(api_name, lib_name)
        types = input_types(api_inputs)
        selection_counts = {}
        api_results = []
        for unused in range(count):
            assembled = assemble_case(api_inputs, selection_counts)
            serialized = {
                name: safe_serialize(info["value"])
                for name, info in assembled.items()
            }
            entry = execute_case(
                run_api,
                target,
                serialized,
                types,
                namespace,
                timeout,
            )
            entry["harness_source"] = source
            api_results.append(entry)
            if entry["函数运行状态"] == "timeout":
                break
        baseline[api_name] = api_results

    save_json(result_dir / "v1_baseline.json", baseline)
    save_json(
        result_dir / "recursion_bugs.json",
        collect_bug_report(baseline, "recursion_bug"),
    )
    save_json(
        result_dir / "timeout_bugs.json",
        collect_bug_report(baseline, "timeout"),
    )


def run_v1(bundle, result_dir, count, timeout):
    record = load_json(bundle / "record.json", {})
    cases = load_json(bundle / "test_cases.json", {})
    if has_path_cases(cases, record):
        replay_validated_path_cases(bundle, timeout)
        materialize_path_baseline(bundle, result_dir)
    else:
        run_v1_legacy(bundle, result_dir, count, timeout)


def normalize_result(value):
    normalized = re.sub(r"0x[0-9a-fA-F]+", "0xXXXX", str(value))
    normalized = re.sub(
        r"(?:[A-Za-z]:)?[^\s'\"<>]*[\\/]\.momo_runtime[\\/]"
        r"worktrees[\\/][^\s/\\'\"<>]+",
        "<WORKTREE>",
        normalized,
    )
    normalized = re.sub(
        r"/(?:private/)?var/folders/[^\s'\"<>]+/T/"
        r"(?:tmp|black_|momo_black_)[^\s/'\"<>]+",
        "<TMPDIR>",
        normalized,
    )
    return normalized


def run_v2_legacy(bundle, result_dir, timeout):
    record = load_json(bundle / "record.json", {})
    inputs = load_json(bundle / "inputs.json", {})
    cases = load_json(bundle / "test_cases.json", {})
    baseline = load_json(result_dir / "v1_baseline.json", {})
    lib_name = record["lib_name"]
    differences = []

    for api_name, baseline_cases in baseline.items():
        api_inputs = inputs.get(api_name, {})
        run_api, target, source = load_run_api(
            api_name,
            lib_name,
            cases.get(api_name),
        )
        if run_api is None:
            differences.append(
                {
                    "api_name": api_name,
                    "case_index": None,
                    "diff_reason": "V2 harness loading failed",
                    "v2_status": "harness_error",
                }
            )
            continue
        namespace = build_eval_globals(api_name, lib_name)
        types = input_types(api_inputs)
        for index, v1_case in enumerate(baseline_cases):
            v2_case = execute_case(
                run_api,
                target,
                v1_case.get("测试输入", {}),
                types,
                namespace,
                timeout,
            )
            v1_status = v1_case.get("函数运行状态")
            v2_status = v2_case.get("函数运行状态")
            v1_output = v1_case.get("函数返回结果")
            v2_output = v2_case.get("函数返回结果")
            v1_phase = v1_case.get("execution_phase")
            v2_phase = v2_case.get("execution_phase")
            v1_call_count = v1_case.get("target_call_count")
            v2_call_count = v2_case.get("target_call_count")
            v1_target_match = v1_case.get("target_matches_api")
            v2_target_match = v2_case.get("target_matches_api")
            if (
                v1_status != v2_status
                or normalize_result(v1_output) != normalize_result(v2_output)
                or v1_phase != v2_phase
                or v1_call_count != v2_call_count
                or v1_target_match != v2_target_match
            ):
                differences.append(
                    {
                        "api_name": api_name,
                        "case_index": index,
                        "inputs": v1_case.get("测试输入", {}),
                        "v1_status": v1_status,
                        "v2_status": v2_status,
                        "v1_output": v1_output,
                        "v2_output": v2_output,
                        "harness_source": source,
                    }
                )
            if v2_status == "timeout":
                break
    save_json(result_dir / "diff_report.json", differences)


def run_v2_path(record, baseline, result_dir, timeout):
    differences = []
    for api_name, baseline_cases in baseline.items():
        for index, v1_case in enumerate(baseline_cases):
            case_data = {
                "case_id": v1_case.get("case_id"),
                "summary": v1_case.get("测试输入", {}).get("summary", ""),
                "code": v1_case.get("test_case_code"),
            }
            v2_case = execute_path_case(
                api_name,
                record["lib_name"],
                case_data,
                timeout,
            )
            v1_status = v1_case.get("函数运行状态")
            v2_status = v2_case.get("函数运行状态")
            v1_output = v1_case.get("函数返回结果")
            v2_output = v2_case.get("函数返回结果")
            v1_phase = v1_case.get("execution_phase")
            v2_phase = v2_case.get("execution_phase")
            v1_call_count = v1_case.get("target_call_count")
            v2_call_count = v2_case.get("target_call_count")
            v1_target_match = v1_case.get("target_matches_api")
            v2_target_match = v2_case.get("target_matches_api")
            if (
                v1_status != v2_status
                or normalize_result(v1_output) != normalize_result(v2_output)
                or v1_phase != v2_phase
                or v1_call_count != v2_call_count
                or v1_target_match != v2_target_match
            ):
                differences.append(
                    {
                        "api_name": api_name,
                        "case_id": v1_case.get("case_id"),
                        "case_index": index,
                        "path_id": v1_case.get("path_id"),
                        "path_constraints": v1_case.get(
                            "path_constraints", []
                        ),
                        "test_case_code": v1_case.get("test_case_code"),
                        "v1_status": v1_status,
                        "v2_status": v2_status,
                        "v1_output": v1_output,
                        "v2_output": v2_output,
                        "v1_phase": v1_phase,
                        "v2_phase": v2_phase,
                        "v1_target_call_count": v1_call_count,
                        "v2_target_call_count": v2_call_count,
                        "v1_target_matches_api": v1_target_match,
                        "v2_target_matches_api": v2_target_match,
                        "harness_source": "path_case",
                    }
                )
    save_json(result_dir / "diff_report.json", differences)


def run_v2(bundle, result_dir, timeout):
    record = load_json(bundle / "record.json", {})
    baseline = load_json(result_dir / "v1_baseline.json", {})
    path_baseline = any(
        isinstance(case, dict) and "test_case_code" in case
        for api_cases in baseline.values()
        if isinstance(api_cases, list)
        for case in api_cases[:1]
    )
    if path_baseline:
        run_v2_path(record, baseline, result_dir, timeout)
    else:
        run_v2_legacy(bundle, result_dir, timeout)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["probe", "v1", "v2"], required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)

    assert_python_version(args.expected_python)
    bundle = Path(args.bundle).resolve()
    result_dir = Path(args.results).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "probe":
        run_probe(bundle, result_dir, args.timeout)
    elif args.mode == "v1":
        run_v1(bundle, result_dir, args.k, args.timeout)
    else:
        run_v2(bundle, result_dir, args.timeout)
    save_json(
        result_dir / ("%s_environment.json" % args.mode),
        {
            "expected_python": args.expected_python,
            "actual_python": exact_python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("test executor failed: %s: %s" % (type(error).__name__, error))
        sys.exit(2)
