import os
import re
from collections import Counter, defaultdict

# 这是一个为 Momo（推测是一个自动程序修复工具）筛选合适 benchmark 的数据分析脚本。


BUGSINPY_ROOT = r"C:\Users\86184\Desktop\Papers\documentation\database\BugsInPy\projects"


def parse_patch(patch_text):
    """Parse a unified diff patch and return structured info"""
    info = {
        "files": [],
        "added_lines": 0,
        "removed_lines": 0,
        "functions_touched": set(),
        "change_types": set(),
    }

    lines = patch_text.split("\n")
    current_file = None
    hunk_funcs = set()

    for line in lines:
        # File header
        file_match = re.match(r"^\+\+\+\s+b/(.+)", line)
        if file_match:
            current_file = file_match.group(1)
            info["files"].append(current_file)

        # Hunk header - extract function/class context
        hunk_match = re.match(r"^@@\s+.+?\s+@@\s+(.+)", line)
        if hunk_match:
            context = hunk_match.group(1)
            for m in re.finditer(r"(def|class)\s+(\w+)", context):
                hunk_funcs.add(f"{m.group(1)} {m.group(2)}")

        # Context line with def/class
        func_match = re.match(r"^\s*(def|class)\s+(\w+)", line)
        if func_match:
            hunk_funcs.add(f"{func_match.group(1)} {func_match.group(2)}")

        # Count additions
        if line.startswith("+") and not line.startswith("+++"):
            info["added_lines"] += 1
            content = line[1:].strip()

            # Classify the change
            if re.match(r"^(def|class)\s+", content):
                info["change_types"].add("new_def")
            elif re.match(r"^(import|from)\s+", content):
                info["change_types"].add("import")
            elif re.match(r"^return\s+", content):
                info["change_types"].add("return_value")
            elif re.match(r"^if\s+|^elif\s+|^else:", content):
                info["change_types"].add("condition")
            elif re.match(r"^try:|^except|^finally:", content):
                info["change_types"].add("exception")
            elif re.match(r"^#|^\"\"\"|^'''", content):
                info["change_types"].add("comment/docstring")
            elif "=" in content and not content.startswith("="):
                info["change_types"].add("assignment")
            elif "re." in content or "regex" in content.lower():
                info["change_types"].add("regex")
            elif content.startswith(".") or content.startswith("self."):
                info["change_types"].add("method_call")
            elif re.match(r"^print|^log|^logger|^logging", content):
                info["change_types"].add("logging")
            elif re.match(r"^raise\s+", content):
                info["change_types"].add("exception")

        # Count removals
        if line.startswith("-") and not line.startswith("---"):
            info["removed_lines"] += 1
            content = line[1:].strip()

            if re.match(r"^(def|class)\s+", content):
                info["change_types"].add("removed_def")
            if re.match(r"^(import|from)\s+", content):
                info["change_types"].add("import")
            if re.match(r"^return\s+", content):
                info["change_types"].add("return_value")
            if re.match(r"^if\s+|^elif\s+|^else:", content):
                info["change_types"].add("condition")
            if re.match(r"^try:|^except|^finally:", content):
                info["change_types"].add("exception")
            if "=" in content and not content.startswith("="):
                info["change_types"].add("assignment")
            if "re." in content or "regex" in content.lower():
                info["change_types"].add("regex")

    info["functions_touched"] = hunk_funcs
    info["change_size"] = info["added_lines"] + info["removed_lines"]

    # Determine change category
    cats = info["change_types"]
    if "regex" in cats:
        info["category"] = "regex/pattern"
    elif "exception" in cats and len(cats) <= 3:
        info["category"] = "exception_handling"
    elif "condition" in cats:
        info["category"] = "logic/condition"
    elif "return_value" in cats:
        info["category"] = "return_value"
    elif "import" in cats and info["added_lines"] <= 3:
        info["category"] = "import"
    elif "new_def" in cats or "removed_def" in cats:
        info["category"] = "structural"
    elif "assignment" in cats:
        info["category"] = "assignment/init"
    elif "logging" in cats:
        info["category"] = "logging"
    elif "comment/docstring" in cats:
        info["category"] = "docstring"
    elif "method_call" in cats:
        info["category"] = "method_call"
    else:
        info["category"] = "other"

    # File type
    py_count = sum(1 for f in info["files"] if f.endswith(".py"))
    if py_count == len(info["files"]) and py_count > 0:
        info["file_type"] = "python_only"
    elif py_count > 0:
        info["file_type"] = "mixed"
    else:
        info["file_type"] = "non_python"

    return info


def analyze_all():
    results = []
    project_stats = {}
    all_categories = Counter()
    all_file_types = Counter()
    change_sizes = []
    func_mentions = Counter()

    for project in sorted(os.listdir(BUGSINPY_ROOT)):
        project_path = os.path.join(BUGSINPY_ROOT, project, "bugs")
        if not os.path.isdir(project_path):
            continue

        proj_cats = Counter()
        proj_bugs = 0

        for bug_dir in sorted(os.listdir(project_path), key=lambda x: int(x) if x.isdigit() else 0):
            bug_path = os.path.join(project_path, bug_dir)
            patch_path = os.path.join(bug_path, "bug_patch.txt")
            if not os.path.isfile(patch_path):
                continue

            with open(patch_path, "r", encoding="utf-8", errors="replace") as f:
                patch_text = f.read()

            info = parse_patch(patch_text)
            info["project"] = project
            info["bug_id"] = bug_dir
            results.append(info)

            all_categories[info["category"]] += 1
            all_file_types[info["file_type"]] += 1
            change_sizes.append(info["change_size"])
            proj_cats[info["category"]] += 1
            proj_bugs += 1
            for f in info["functions_touched"]:
                func_mentions[f] += 1

        if proj_bugs > 0:
            project_stats[project] = {"total": proj_bugs, "categories": proj_cats}

    # ── Print Report ──
    print("=" * 70)
    print("  BugsInPy Bug Analysis Report (from bug_patch.txt)")
    print(f"  Total bugs analyzed: {len(results)}")
    print("=" * 70)

    print("\n── Bug Category Distribution ──")
    print(f"  {'Category':<22} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*34}")
    for cat, count in all_categories.most_common():
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:<22} {count:>6}  {pct:>5.1f}%  {bar}")

    print("\n── File Type Distribution ──")
    for ft, count in all_file_types.most_common():
        print(f"  {ft:<22} {count:>6}  {count/len(results)*100:>5.1f}%")

    print(f"\n── Change Size ──")
    change_sizes.sort()
    print(f"  Min: {change_sizes[0]} lines  |  Max: {change_sizes[-1]} lines")
    print(f"  Mean: {sum(change_sizes)/len(change_sizes):.1f} lines  |  Median: {change_sizes[len(change_sizes)//2]} lines")

    print(f"\n── Per-Project Summary ──")
    print(f"  {'Project':<16} {'Bugs':>5}  Top Categories")
    print(f"  {'-'*50}")
    for proj in sorted(project_stats.keys(), key=lambda p: project_stats[p]["total"], reverse=True):
        ps = project_stats[proj]
        top = ps["categories"].most_common(3)
        top_str = ", ".join(f"{c}({n})" for c, n in top)
        print(f"  {proj:<16} {ps['total']:>5}  {top_str}")

    print(f"\n── Most Modified Functions (top 15) ──")
    for func, count in func_mentions.most_common(15):
        bar = "█" * count
        print(f"  {func:<40} {count:>3}  {bar}")

    print(f"\n── Momo-Relevant Bugs ──")
    # Bugs suitable for Momo: structural changes, logic/condition, return_value, assignment
    momo_cats = {"logic/condition", "return_value", "assignment/init", "structural", "method_call"}
    momo_count = sum(all_categories[c] for c in momo_cats)
    non_momo = {"regex/pattern", "import", "docstring", "logging"}
    non_count = sum(all_categories[c] for c in non_momo)
    other_count = len(results) - momo_count - non_count

    print(f"  Suitable (logic, return, assignment, structural, method_call): {momo_count} ({momo_count/len(results)*100:.1f}%)")
    print(f"  Less suitable (regex, import, docstring, logging):            {non_count} ({non_count/len(results)*100:.1f}%)")
    print(f"  Other/uncategorized:                                           {other_count} ({other_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    analyze_all()
