#!/usr/bin/env python3
"""Calculate detection and target-API coverage metrics for Momo results.

The script has no third-party dependencies. Coverage prefers native SlipCover
JSON reports named ``*slipcover*.json`` in sample result directories. If none
exist, it reconstructs target-API coverage from saved baseline traces and the
exact bug revision in ``documentation/dl_lib/<project>``.
"""

import argparse
import ast
import dis
import json
import re
import subprocess
import sys
import types
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL = Path(__file__).with_name("实验数据.xlsx")
DEFAULT_RESULTS = ROOT / "documentation" / "results"
DEFAULT_REPOS = ROOT / "documentation" / "dl_lib"
XML_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {
    "r": "http://schemas.openxmlformats.org/package/2006/relationships"
}


def column_name(cell_reference):
    return re.match(r"[A-Z]+", cell_reference).group(0)


def read_xlsx_rows(path):
    """Read the first worksheet containing sample/project using stdlib only."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(".//x:t", XML_NS))
                for item in root.findall("x:si", XML_NS)
            ]

        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall("r:Relationship", REL_NS)
        }

        for sheet in workbook.findall(".//x:sheet", XML_NS):
            relation_id = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = targets[relation_id].lstrip("/")
            worksheet_path = target if target.startswith("xl/") else "xl/" + target
            worksheet = ElementTree.fromstring(archive.read(worksheet_path))
            rows = []
            for row in worksheet.findall(".//x:sheetData/x:row", XML_NS):
                values = {}
                for cell in row.findall("x:c", XML_NS):
                    value_node = cell.find("x:v", XML_NS)
                    value = "" if value_node is None else (value_node.text or "")
                    if cell.attrib.get("t") == "s" and value:
                        value = shared[int(value)]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(
                            node.text or "" for node in cell.findall(".//x:t", XML_NS)
                        )
                    values[column_name(cell.attrib["r"])] = value
                rows.append(values)

            if not rows:
                continue
            header = {str(value).strip().lower(): column for column, value in rows[0].items()}
            if "sample" not in header or "project" not in header:
                continue
            result = []
            for row in rows[1:]:
                sample = str(row.get(header["sample"], "")).strip()
                project = str(row.get(header["project"], "")).strip()
                if sample and project:
                    result.append((sample, project))
            return result

    raise ValueError("Excel 中未找到同时包含 sample 和 project 列的工作表")


def sample_directory(results_dir, sample, project):
    candidates = [
        results_dir / project / sample,
        results_dir / project / sample.replace("__", "_"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[-1]


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def is_nonempty_json(path):
    data = load_json(path)
    if data is None:
        return False
    if isinstance(data, (list, dict, str)):
        return bool(data)
    return True


def calculate_detection_metrics(tasks, results_dir):
    recall_count = 0
    precision_count = 0
    missing_samples = []
    invalid_reports = []

    for sample, project in tasks:
        directory = sample_directory(results_dir, sample, project)
        reports = sorted(directory.glob("*diff_report.json")) if directory.is_dir() else []
        if not directory.is_dir():
            missing_samples.append("%s/%s" % (project, sample))
        if reports:
            recall_count += 1
            if any(is_nonempty_json(report) for report in reports):
                precision_count += 1
            elif any(load_json(report) is None for report in reports):
                invalid_reports.extend(str(report) for report in reports if load_json(report) is None)

    task_count = len(tasks)
    recall = recall_count / task_count if task_count else 0.0
    precision = precision_count / recall_count if recall_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "task_count": task_count,
        "recall_count": recall_count,
        "precision_count": precision_count,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "missing_samples": missing_samples,
        "invalid_reports": invalid_reports,
    }


def find_slipcover_reports(tasks, results_dir):
    reports = []
    for sample, project in tasks:
        directory = sample_directory(results_dir, sample, project)
        if directory.is_dir():
            reports.extend(sorted(directory.glob("*slipcover*.json")))
    return reports


def aggregate_slipcover(reports):
    files = defaultdict(
        lambda: {
            "executed_lines": set(),
            "possible_lines": set(),
            "executed_branches": set(),
            "possible_branches": set(),
        }
    )
    valid_reports = 0
    for report in reports:
        data = load_json(report)
        if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
            continue
        valid_reports += 1
        for filename, entry in data["files"].items():
            if not isinstance(entry, dict):
                continue
            aggregate = files[filename]
            executed_lines = set(entry.get("executed_lines", []))
            missing_lines = set(entry.get("missing_lines", []))
            executed_branches = {
                tuple(branch) for branch in entry.get("executed_branches", [])
            }
            missing_branches = {
                tuple(branch) for branch in entry.get("missing_branches", [])
            }
            aggregate["executed_lines"].update(executed_lines)
            aggregate["possible_lines"].update(executed_lines | missing_lines)
            aggregate["executed_branches"].update(executed_branches)
            aggregate["possible_branches"].update(
                executed_branches | missing_branches
            )

    covered_lines = sum(len(item["executed_lines"]) for item in files.values())
    total_lines = sum(len(item["possible_lines"]) for item in files.values())
    covered_branches = sum(
        len(item["executed_branches"]) for item in files.values()
    )
    total_branches = sum(len(item["possible_branches"]) for item in files.values())
    return coverage_result(
        covered_lines,
        total_lines - covered_lines,
        covered_branches,
        total_branches - covered_branches,
        "slipcover",
        {"reports": valid_reports, "files": len(files)},
    )


def git_output(repo, arguments):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo)] + arguments,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def repository_for(project, repos_dir):
    direct = repos_dir / project
    if (direct / ".git").exists():
        return direct
    lowered = project.lower()
    for candidate in repos_dir.iterdir() if repos_dir.is_dir() else []:
        if candidate.name.lower() == lowered and (candidate / ".git").exists():
            return candidate
    return None


def match_git_path(trace_path, tracked_files):
    normalized = str(trace_path).replace("\\", "/")
    matches = [name for name in tracked_files if normalized.endswith("/" + name)]
    if not matches:
        return None
    return max(matches, key=len)


def executable_lines(source, filename, start_line, end_line):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            root_code = compile(source, filename, "exec")
    except (SyntaxError, ValueError):
        return set()
    result = set()
    pending = [root_code]
    while pending:
        code = pending.pop()
        if hasattr(code, "co_lines"):
            result.update(
                line
                for _, _, line in code.co_lines()
                if line is not None and start_line <= line <= end_line
            )
        else:
            result.update(
                line
                for _, line in dis.findlinestarts(code)
                if start_line <= line <= end_line
            )
        pending.extend(item for item in code.co_consts if isinstance(item, types.CodeType))
    return result


def statement_lines(nodes):
    result = set()
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.stmt) and hasattr(child, "lineno"):
                result.add(child.lineno)
    return result


def target_execution_scope(source, start_line, end_line, observed_lines):
    """Return executable lines and AST ranges relevant to one API invocation.

    inspect.getsourcelines(class) returns the entire class. A class call does
    not execute every method in that class, so for class targets we include
    only methods that the saved target trace actually entered.
    """
    executable = executable_lines(source, "<target>", start_line, end_line)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return executable - {start_line}, [(start_line, end_line)]

    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and start_line <= node.lineno
        and getattr(node, "end_lineno", node.lineno) <= end_line
    ]
    if not candidates:
        return executable - {start_line}, [(start_line, end_line)]
    target = max(
        candidates,
        key=lambda node: getattr(node, "end_lineno", node.lineno) - node.lineno,
    )

    if not isinstance(target, ast.ClassDef):
        body_start = min(
            (node.lineno for node in target.body if hasattr(node, "lineno")),
            default=target.lineno + 1,
        )
        scope = (body_start, getattr(target, "end_lineno", end_line))
        return {
            line for line in executable if scope[0] <= line <= scope[1]
        }, [scope]

    method_ranges = []
    for node in ast.walk(target):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body_start = min(
            (item.lineno for item in node.body if hasattr(item, "lineno")),
            default=node.lineno + 1,
        )
        method_end = getattr(node, "end_lineno", body_start)
        method_lines = {
            line for line in executable if body_start <= line <= method_end
        }
        if method_lines & observed_lines:
            method_ranges.append((body_start, method_end))

    if not method_ranges:
        return executable & observed_lines, []
    scoped = {
        line
        for line in executable
        if any(lower <= line <= upper for lower, upper in method_ranges)
    }
    return scoped, method_ranges


def statement_successors(tree):
    """Map statements to the next statement in the same lexical block."""
    successors = {}
    for parent in ast.walk(tree):
        for _, value in ast.iter_fields(parent):
            if not isinstance(value, list) or not value:
                continue
            if not all(isinstance(item, ast.stmt) for item in value):
                continue
            for current, following in zip(value, value[1:]):
                successors[id(current)] = following.lineno
    return successors


def source_branches(source, start_line, end_line, case_traces, scope_ranges):
    """Infer source-level branch outcomes from per-case executed-line sets.

    Only outcomes distinguishable through line events are included. Same-line
    Boolean short-circuit and conditional-expression outcomes are omitted.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return set(), set()

    possible = set()
    covered = set()
    successors = statement_successors(tree)

    def in_scope(node):
        return (
            start_line <= node.lineno <= end_line
            and any(lower <= node.lineno <= upper for lower, upper in scope_ranges)
        )

    def traces():
        for trace in case_traces:
            if isinstance(trace, dict):
                yield trace.get("lines", set()), bool(trace.get("completed"))
            else:
                yield trace, False

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try)
        ) and not (hasattr(ast, "Match") and isinstance(node, ast.Match)):
            continue
        if not in_scope(node):
            continue

        if isinstance(node, ast.If):
            true_lines = statement_lines(node.body) - {node.lineno}
            false_lines = statement_lines(node.orelse) - {node.lineno}
            if not true_lines:
                continue
            possible.update({(node.lineno, "if:true"), (node.lineno, "if:false")})
            for lines, completed in traces():
                if node.lineno not in lines:
                    continue
                if lines & true_lines:
                    covered.add((node.lineno, "if:true"))
                if false_lines:
                    if lines & false_lines:
                        covered.add((node.lineno, "if:false"))
                elif not (lines & true_lines) and completed:
                    covered.add((node.lineno, "if:false"))
            continue

        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            kind = "while" if isinstance(node, ast.While) else "for"
            body_lines = statement_lines(node.body) - {node.lineno}
            exit_lines = statement_lines(node.orelse) - {node.lineno}
            successor = successors.get(id(node))
            if successor is not None:
                exit_lines.add(successor)
            if not body_lines:
                continue
            possible.update(
                {(node.lineno, kind + ":body"), (node.lineno, kind + ":exit")}
            )
            for lines, completed in traces():
                if node.lineno not in lines:
                    continue
                if lines & body_lines:
                    covered.add((node.lineno, kind + ":body"))
                if lines & exit_lines or (completed and not (lines & body_lines)):
                    covered.add((node.lineno, kind + ":exit"))
            continue

        if isinstance(node, ast.Try):
            body_lines = statement_lines(node.body) - {node.lineno}
            handler_lines = [
                statement_lines(handler.body) | {handler.lineno}
                for handler in node.handlers
            ]
            if not body_lines or not handler_lines:
                continue
            normal_lines = statement_lines(node.orelse)
            successor = successors.get(id(node))
            if successor is not None:
                normal_lines.add(successor)
            possible.add((node.lineno, "try:normal"))
            possible.update(
                (node.lineno, "try:except_%d" % index)
                for index in range(len(handler_lines))
            )
            for lines, completed in traces():
                if not (lines & body_lines):
                    continue
                matched_handler = False
                for index, branch_lines in enumerate(handler_lines):
                    if lines & branch_lines:
                        matched_handler = True
                        covered.add((node.lineno, "try:except_%d" % index))
                if lines & normal_lines or (completed and not matched_handler):
                    covered.add((node.lineno, "try:normal"))
            continue

        if hasattr(ast, "Match") and isinstance(node, ast.Match):
            case_lines = [statement_lines(case.body) for case in node.cases]
            if len(case_lines) < 2:
                continue
            for index in range(len(case_lines)):
                possible.add((node.lineno, "match:case_%d" % index))
            for lines, _ in traces():
                for index, branch_lines in enumerate(case_lines):
                    if lines & branch_lines:
                        covered.add((node.lineno, "match:case_%d" % index))

    return covered, possible


def iter_baseline_cases(data):
    if not isinstance(data, dict):
        return
    for value in data.values():
        if not isinstance(value, list):
            continue
        for case in value:
            if isinstance(case, dict):
                yield case


def calculate_path_coverage(tasks, results_dir):
    total = 0
    validated = 0
    samples = 0
    for sample, project in tasks:
        directory = sample_directory(results_dir, sample, project)
        manifests = sorted(directory.glob("run_manifest.json"))
        if not manifests:
            continue
        manifest = load_json(manifests[0])
        metadata = manifest.get("metadata", {}) if isinstance(manifest, dict) else {}
        pruning = metadata.get("path_case_pruning", {})
        sample_total = pruning.get("total")
        sample_validated = pruning.get("validated")
        if isinstance(sample_total, int) and isinstance(sample_validated, int):
            total += sample_total
            validated += sample_validated
            samples += 1
    return {
        "validated_paths": validated,
        "static_paths": total,
        "path_coverage": ratio(validated, total),
        "samples": samples,
    }


def aggregate_trace_coverage(tasks, results_dir, repos_dir):
    targets = {}
    skipped = defaultdict(int)
    tree_cache = {}
    source_cache = {}

    for sample, project in tasks:
        directory = sample_directory(results_dir, sample, project)
        manifests = sorted(directory.glob("run_manifest.json"))
        baselines = sorted(directory.glob("*_v1_baseline.json"))
        if not manifests or not baselines:
            skipped["missing_manifest_or_baseline"] += 1
            continue
        manifest = load_json(manifests[0])
        baseline = load_json(baselines[0])
        record = manifest.get("record", {}) if isinstance(manifest, dict) else {}
        commit = str(record.get("bug_version", "")).strip()
        repo = repository_for(project, repos_dir)
        if not commit or repo is None:
            skipped["missing_repository_or_commit"] += 1
            continue

        cache_key = (str(repo), commit)
        if cache_key not in tree_cache:
            listing = git_output(repo, ["ls-tree", "-r", "--name-only", commit])
            tree_cache[cache_key] = listing.splitlines() if listing else []

        found_trace = False
        for case in iter_baseline_cases(baseline):
            trace = case.get("target_trace")
            if not isinstance(trace, dict):
                continue
            trace_file = trace.get("file")
            start = trace.get("start_line")
            end = trace.get("end_line")
            lines = trace.get("executed_lines")
            if not trace_file or not isinstance(start, int) or not isinstance(end, int):
                continue
            if not isinstance(lines, list):
                continue
            git_path = match_git_path(trace_file, tree_cache[cache_key])
            if git_path is None:
                skipped["source_path_not_found"] += 1
                continue
            found_trace = True
            target_key = (str(repo), commit, git_path, start, end)
            target = targets.setdefault(
                target_key, {"covered_lines": set(), "case_lines": []}
            )
            observed = {line for line in lines if isinstance(line, int)}
            target["covered_lines"].update(observed)
            target["case_lines"].append(
                {
                    "lines": observed,
                    "completed": bool(case.get("target_completed")),
                }
            )
        if not found_trace:
            skipped["no_usable_target_trace"] += 1

    covered_lines = set()
    possible_lines = set()
    covered_branches = set()
    possible_branches = set()
    resolved_targets = 0
    for key, target in targets.items():
        repo_name, commit, git_path, start, end = key
        source_key = (repo_name, commit, git_path)
        if source_key not in source_cache:
            source_cache[source_key] = git_output(
                Path(repo_name), ["show", "%s:%s" % (commit, git_path)]
            )
        source = source_cache[source_key]
        if source is None:
            skipped["source_content_not_found"] += 1
            continue
        observed_lines = target["covered_lines"]
        executable, scope_ranges = target_execution_scope(
            source, start, end, observed_lines
        )
        if not executable:
            skipped["source_not_compilable"] += 1
            continue
        resolved_targets += 1
        identity = (repo_name, commit, git_path, start, end)
        possible_lines.update((identity, line) for line in executable)
        covered_lines.update(
            (identity, line)
            for line in target["covered_lines"]
            if line in executable
        )
        branch_covered, branch_possible = source_branches(
            source, start, end, target["case_lines"], scope_ranges
        )
        covered_branches.update((identity, branch) for branch in branch_covered)
        possible_branches.update((identity, branch) for branch in branch_possible)

    return coverage_result(
        len(covered_lines),
        len(possible_lines - covered_lines),
        len(covered_branches),
        len(possible_branches - covered_branches),
        "saved-target-invocation-trace-proxy",
        {
            "resolved_targets": resolved_targets,
            "skipped": dict(sorted(skipped.items())),
            "note": (
                "Function coverage uses the function body; class-call coverage "
                "uses only methods entered by that invocation. Source branch "
                "outcomes are reconstructed from the exact bug commit and each "
                "case's executed-line set."
            ),
        },
    )


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def coverage_result(
    covered_lines,
    missing_lines,
    covered_branches,
    missing_branches,
    method,
    details,
):
    line_total = covered_lines + missing_lines
    branch_total = covered_branches + missing_branches
    combined_total = line_total + branch_total
    return {
        "method": method,
        "covered_lines": covered_lines,
        "missing_lines": missing_lines,
        "covered_branches": covered_branches,
        "missing_branches": missing_branches,
        "line_coverage": ratio(covered_lines, line_total),
        "branch_coverage": ratio(covered_branches, branch_total),
        "combined_coverage": ratio(
            covered_lines + covered_branches, combined_total
        ),
        "details": details,
    }


def percent(value):
    return "N/A" if value is None else "%.2f%%" % (value * 100.0)


def print_results(detection, path_coverage, coverage):
    print("=== Detection Metrics ===")
    print("task_cnt      : %d" % detection["task_count"])
    print("recall_cnt    : %d" % detection["recall_count"])
    print("precision_cnt : %d" % detection["precision_count"])
    print("Recall        : %s" % percent(detection["recall"]))
    print("Precision     : %s" % percent(detection["precision"]))
    print("F1            : %s" % percent(detection["f1"]))
    print()
    print("=== AST Path Coverage ===")
    print(
        "paths         : %d validated, %d generated"
        % (path_coverage["validated_paths"], path_coverage["static_paths"])
    )
    print("Path Coverage : %s" % percent(path_coverage["path_coverage"]))
    print()
    print("=== Coverage Metrics ===")
    print("method             : %s" % coverage["method"])
    print(
        "lines              : %d covered, %d missing"
        % (coverage["covered_lines"], coverage["missing_lines"])
    )
    print(
        "branches           : %d covered, %d missing"
        % (coverage["covered_branches"], coverage["missing_branches"])
    )
    print("Line Coverage      : %s" % percent(coverage["line_coverage"]))
    print("Branch Coverage    : %s" % percent(coverage["branch_coverage"]))
    print("Combined Coverage  : %s" % percent(coverage["combined_coverage"]))
    details = coverage.get("details", {})
    if details.get("skipped"):
        summary = ", ".join(
            "%s=%s" % item for item in details["skipped"].items()
        )
        print("coverage gaps      : %s" % summary)
    if detection["missing_samples"]:
        print("missing samples    : %d" % len(detection["missing_samples"]))
    if detection["invalid_reports"]:
        print("invalid diff JSON  : %d" % len(detection["invalid_reports"]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Recall, Precision, F1 and code coverage."
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--repos-dir", type=Path, default=DEFAULT_REPOS)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally save all metrics and diagnostics as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = read_xlsx_rows(args.excel)
    if not tasks:
        raise SystemExit("Excel 中没有有效的 sample + project 数据项")
    if len(tasks) != len(set(tasks)):
        raise SystemExit("Excel 中存在重复的 sample + project 数据项")

    detection = calculate_detection_metrics(tasks, args.results_dir)
    path_coverage = calculate_path_coverage(tasks, args.results_dir)
    slipcover_reports = find_slipcover_reports(tasks, args.results_dir)
    if slipcover_reports:
        coverage = aggregate_slipcover(slipcover_reports)
    else:
        coverage = aggregate_trace_coverage(tasks, args.results_dir, args.repos_dir)

    print_results(detection, path_coverage, coverage)
    if args.json_output:
        output = {
            "detection": detection,
            "path_coverage": path_coverage,
            "coverage": coverage,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nJSON written to: %s" % args.json_output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print("错误: %s" % error, file=sys.stderr)
        raise SystemExit(1)
