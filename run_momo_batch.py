import argparse
import ctypes
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from test_environment import (
    DEFAULT_INDEX_URL,
    StrictTestEnvironment,
    TestEnvironmentError,
    normalize_python_version,
    prepare_requirements,
)


ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = ROOT / "documentation" / "database"
RESULTS_DIR = ROOT / "documentation" / "results"
MOMO_DIR = ROOT / "Momo_test"
DL_LIB_DIR = ROOT / "documentation" / "dl_lib"
BUGSINPY_DIR = DATABASE_DIR / "BugsInPy" / "projects"
SKIPPED_DIR = DATABASE_DIR / "skipped_entries"
RUNTIME_DIR = ROOT / ".momo_runtime"
JOERN_WORKSPACE = RUNTIME_DIR / "workspace"

JOERN_EXE = ROOT / "joern-cli" / ("joern.bat" if os.name == "nt" else "joern")
if not JOERN_EXE.exists():
    JOERN_EXE = ROOT / "joern-cli" / "bin" / (
        "joern-cli.bat" if os.name == "nt" else "joern-cli"
    )

LIB_FILE = "black.txt"
LIB_GITNAME = "black"
LIB_NAME = "black"
START = 0
END = 1
DEFAULT_K = 1
DEFAULT_MAX_REPAIR_ROUNDS = 8
BASE_PYTHON = Path(os.path.abspath(sys.executable))

# Disabled by default because both options mutate the selected Python environment.
INSTALL_DEPENDENCIES = False
INSTALL_TARGET_PACKAGE = False


INTERMEDIATE_PATTERNS = {
    "api_guards": ["{lib}_api_guards.json"],
    "api_input": ["{lib}_default_inputs_*.json", "{lib}_inputs_*.json"],
    "api_src_code": ["{lib}_api_sources.json"],
    "arg_boundary": ["cut_{lib}_boundary_*.json", "{lib}_boundary_*.json"],
    "arg_combinations": [
        "{lib}_cut_combinations_*.json",
        "{lib}_combinations_*.json",
    ],
    "arg_space": ["{lib}_arg_space_*.json"],
    "error_combinations": ["error_{lib}_combinations.json"],
    "conditions": ["{lib}_conditions.json"],
    "test_cases": ["{lib}_case_*.json"],
}

RESULT_PATTERNS = [
    "{lib}_v1_baseline*.json",
    "{lib}_diff_report*.json",
    "{lib}_recursion_bugs*.json",
    "{lib}_timeout_bugs*.json",
    "{lib}_crash_bugs*.json",
]


def is_admin():
    if os.name != "nt":
        return True
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def relaunch_as_admin():
    script = Path(__file__).resolve()
    args = " ".join([f'"{script}"', *[f'"{arg}"' for arg in sys.argv[1:]]])
    result = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, args, str(MOMO_DIR), 1
    )
    if result <= 32:
        raise RuntimeError(f"管理员权限启动失败，ShellExecuteW 返回值：{result}")
    print("已请求管理员权限重新启动脚本，当前非管理员进程退出。")
    sys.exit(0)


def _display_command(command):
    if isinstance(command, str):
        return command
    try:
        return shlex.join(str(part) for part in command)
    except AttributeError:
        return " ".join(shlex.quote(str(part)) for part in command)


def run(command, cwd=None, env=None, check=True, timeout=None):
    print(f"\n>>> {_display_command(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        shell=isinstance(command, str),
        check=check,
        env=env,
        timeout=timeout,
    )


def run_no_check(command, cwd=None, env=None, timeout=None):
    return run(command, cwd=cwd, env=env, check=False, timeout=timeout).returncode == 0


def run_python(script, args=None, cwd=None, env=None, python_executable=BASE_PYTHON):
    command = [str(python_executable), "-u", str(script)]
    if args:
        command.extend(str(arg) for arg in args)
    run(command, cwd=cwd, env=env)


def make_base_env():
    cache_dir = RUNTIME_DIR / "cache"
    config_dir = RUNTIME_DIR / "config"
    joern_home = RUNTIME_DIR / "joern"
    for path in (
        RUNTIME_DIR,
        cache_dir,
        config_dir,
        joern_home,
        JOERN_WORKSPACE,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return {
        **os.environ,
        "MOMO_ROOT": str(ROOT),
        "MOMO_RUNTIME_DIR": str(RUNTIME_DIR),
        "JOERN_PATH": str(JOERN_EXE),
        "XDG_CACHE_HOME": str(cache_dir),
        "XDG_CONFIG_HOME": str(config_dir),
        "JOERN_HOME": str(joern_home),
    }


def make_runtime_env(repo_dir, lib_name, lib_gitname, joern_project):
    python_paths = [
        str(repo_dir),
        str(repo_dir / "src"),
        str(repo_dir / "lib"),
        str(MOMO_DIR),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)

    env = make_base_env()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "MOMO_LIB_NAME": lib_name,
            "MOMO_LIB_GITNAME": lib_gitname,
            "MOMO_TARGET_REPO": str(Path(repo_dir).resolve()),
            "MOMO_JOERN_PROJECT": joern_project,
            "MOMO_USE_SOURCE_RESOLVER": "1",
        }
    )
    return env


def parse_lib_file(path):
    text = Path(path).read_text(encoding="utf-8")
    records = []

    for block_index, block in enumerate(
        re.split(r"^--------\s*$", text, flags=re.MULTILINE), start=1
    ):
        lines = block.splitlines()
        if not any(line.strip() for line in lines):
            continue

        bug_id = None
        python_version = ""
        bug_hash = None
        fix_hash = None
        apis = []
        in_api_section = False
        invalid = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = re.match(r"^bug_id:\s*(.+)$", line)
            if match:
                try:
                    bug_id = int(match.group(1).strip())
                except ValueError:
                    print(f"警告：第 {block_index} 块的 bug_id 非整数，已跳过。")
                    invalid = True
                in_api_section = False
                continue

            match = re.match(r"^python_version:\s*(.+)$", line)
            if match:
                python_version = match.group(1).strip()
                in_api_section = False
                continue

            match = re.match(r"^buggy:\s*(.+)$", line)
            if match:
                bug_hash = match.group(1).strip()
                in_api_section = False
                continue

            match = re.match(r"^fixed:\s*(.+)$", line)
            if match:
                fix_hash = match.group(1).strip()
                in_api_section = False
                continue

            if re.match(r"^bug_api:\s*$", line):
                in_api_section = True
                continue

            if in_api_section and stripped != "(none)":
                apis.append(stripped)

        if invalid or bug_id is None or not bug_hash or not fix_hash or not apis:
            print(
                "警告：跳过字段不完整的条目 "
                f"(bug_id={bug_id}, buggy={bug_hash}, fixed={fix_hash}, apis={len(apis)})"
            )
            continue

        records.append(
            {
                "bug_id": bug_id,
                "python_version": python_version,
                "bug_version": bug_hash,
                "fix_version": fix_hash,
                "bug_api": apis,
            }
        )

    return records


def parse_shell_metadata(path):
    data = {}
    path = Path(path)
    if not path.exists():
        return data
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.isidentifier():
            data[key] = value.strip().strip("\"'")
    return data


def bugsinpy_record_metadata(lib_gitname, record):
    bug_dir = BUGSINPY_DIR / lib_gitname / "bugs" / str(record["bug_id"])
    bug_info = parse_shell_metadata(bug_dir / "bug.info")
    required_version = normalize_python_version(record["python_version"])
    info_version = bug_info.get("python_version")
    if info_version and normalize_python_version(info_version) != required_version:
        raise TestEnvironmentError(
            "LIB_FILE python_version=%s differs from bug.info python_version=%s"
            % (required_version, info_version)
        )
    return {
        "bug_dir": bug_dir,
        "required_python": required_version,
        "pythonpath": bug_info.get("pythonpath", ""),
        "test_file": bug_info.get("test_file", ""),
        "requirements": bug_dir / "requirements.txt",
        "setup": bug_dir / "setup.sh",
        "run_test": bug_dir / "run_test.sh",
    }


def _merge_json_files(directory, pattern):
    merged = {}
    for path in sorted(Path(directory).glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("invalid JSON artifact %s: %s" % (path, error))
        if isinstance(data, dict):
            merged.update(data)
    return merged


def prepare_test_bundle(run_dir, lib_name, record):
    bundle_dir = Path(run_dir) / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    api_names = []
    for definition in record["bug_api"]:
        api_names.append(definition.split("(", 1)[0].strip())

    bundle_record = {
        "lib_name": lib_name,
        "bug_id": record["bug_id"],
        "required_python": record["python_version"],
        "bug_version": record["bug_version"],
        "fix_version": record["fix_version"],
        "api_names": api_names,
    }
    (bundle_dir / "record.json").write_text(
        json.dumps(bundle_record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    inputs = _merge_json_files(
        ROOT / "documentation" / "api_input",
        f"{lib_name}_inputs_*.json",
    )
    test_cases = _merge_json_files(
        ROOT / "documentation" / "test_cases",
        f"{lib_name}_case_*.json",
    )
    (bundle_dir / "inputs.json").write_text(
        json.dumps(inputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (bundle_dir / "test_cases.json").write_text(
        json.dumps(test_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    path_case_apis = {
        name
        for name, api_cases in test_cases.items()
        if isinstance(api_cases, list)
        and api_cases
        and isinstance(api_cases[0], dict)
        and api_cases[0].get("schema_version") == 2
    }
    if path_case_apis:
        missing_cases = [name for name in api_names if name not in path_case_apis]
        if missing_cases:
            raise RuntimeError(
                "test bundle is missing path cases for: %s" % missing_cases
            )
    else:
        missing_inputs = [name for name in api_names if name not in inputs]
        if missing_inputs:
            raise RuntimeError(
                "test bundle is missing inputs for: %s" % missing_inputs
            )
    return bundle_dir


def publish_executor_results(lib_name, executor_results):
    executor_results = Path(executor_results)
    mapping = {
        "v1_baseline.json": f"{lib_name}_v1_baseline.json",
        "diff_report.json": f"{lib_name}_diff_report.json",
        "recursion_bugs.json": f"{lib_name}_recursion_bugs.json",
        "timeout_bugs.json": f"{lib_name}_timeout_bugs.json",
    }
    for source_name, destination_name in mapping.items():
        source = executor_results / source_name
        if not source.exists():
            if source_name in ("recursion_bugs.json", "timeout_bugs.json"):
                source.write_text("{}", encoding="utf-8")
            else:
                raise RuntimeError("test executor did not produce %s" % source)
        shutil.copy2(source, RESULTS_DIR / destination_name)


def _is_prerelease_version(line):
    return bool(
        re.search(r"==[^;]*\.dev\d+", line.lower())
        or re.search(
            r"==[^;]*(?:alpha|beta|rc)\d*", line.lower(), re.IGNORECASE
        )
    )


def install_requirements(bug_id, lib_gitname, python_executable):
    req_path = BUGSINPY_DIR / lib_gitname / "bugs" / str(bug_id) / "requirements.txt"
    if not req_path.exists():
        print(f"未找到 requirements.txt: {req_path}，跳过依赖安装。")
        return

    print(f"安装依赖: {req_path}")
    known_broken = {
        "accessify",
        "funcsigs",
        "pathlib2",
        "scandir",
        "functools32",
        "typing",
    }
    raw = req_path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("utf-16-le")

    filtered = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            filtered.append(line)
            continue
        if stripped.startswith("-e"):
            print(f"跳过可编辑安装: {stripped}")
            continue
        pkg_name = re.split(r"==|>=|<=|>|<", stripped, maxsplit=1)[0].strip()
        if pkg_name.lower() in known_broken:
            print(f"跳过已知不兼容包: {stripped}")
        elif _is_prerelease_version(stripped):
            print(f"跳过预发布版本: {stripped}")
        else:
            filtered.append(line)

    filtered_text = "\n".join(filtered)
    if not filtered_text.strip():
        print("过滤后无有效依赖，跳过安装。")
        return

    temp_dir = RUNTIME_DIR / "requirements"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_req = temp_dir / f"{lib_gitname}_{bug_id}.txt"
    temp_req.write_text(filtered_text, encoding="utf-8")
    try:
        if not run_no_check(
            [str(python_executable), "-m", "pip", "install", "-r", str(temp_req)]
        ):
            print("警告：部分依赖安装失败，继续使用当前环境。")
    finally:
        if temp_req.exists():
            temp_req.unlink()


def install_target_package(repo_dir, env, python_executable):
    normal = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        ".",
    ]
    editable = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "-e",
        ".",
    ]
    if run_no_check(normal, cwd=repo_dir, env=env):
        return True
    print("普通安装失败，尝试 editable 安装。")
    if run_no_check(editable, cwd=repo_dir, env=env):
        return True
    print("警告：目标库安装失败，将继续使用 PYTHONPATH。")
    return False


def _venv_python(venv_dir):
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _has_runner_dependencies(python_executable):
    result = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import openai; import yaml; import psutil; import json_repair",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def ensure_runner_python(python_executable, bootstrap=True):
    if _has_runner_dependencies(python_executable):
        return Path(os.path.abspath(str(python_executable)))
    if not bootstrap:
        raise RuntimeError(
            "运行解释器缺少核心依赖。请执行 "
            f"{python_executable} -m pip install -r "
            f"{MOMO_DIR / 'requirements-runner.txt'}"
        )

    version = subprocess.check_output(
        [
            str(python_executable),
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ],
        text=True,
    ).strip()
    venv_dir = RUNTIME_DIR / f"runner-py{version}"
    worker_python = _venv_python(venv_dir)
    if not worker_python.exists():
        print(f"创建隔离 runner 环境: {venv_dir}")
        run([str(python_executable), "-m", "venv", str(venv_dir)])

    requirements = MOMO_DIR / "requirements-runner.txt"
    print(f"安装 runner 依赖: {requirements}")
    run(
        [
            str(worker_python),
            "-m",
            "pip",
            "install",
            "--timeout",
            "30",
            "--retries",
            "2",
            "-r",
            str(requirements),
        ],
        timeout=600,
    )
    if not _has_runner_dependencies(worker_python):
        raise RuntimeError(f"runner 依赖安装后仍不可导入: {worker_python}")
    return Path(os.path.abspath(str(worker_python)))


def write_api_defs(lib_name, apis):
    api_dir = ROOT / "documentation" / "lib_api"
    api_dir.mkdir(parents=True, exist_ok=True)
    api_path = api_dir / f"{lib_name}_APIdef.txt"
    api_path.write_text("\n".join(apis) + "\n", encoding="utf-8")
    print(f"Wrote {api_path}")


class JoernShell:
    def __init__(self, joern_path):
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [str(joern_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="gbk" if os.name == "nt" else "utf-8",
            errors="replace",
            shell=False,
            cwd=RUNTIME_DIR,
            env=make_base_env(),
        )
        self.output_queue = queue.Queue()
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self):
        try:
            for line in iter(self.process.stdout.readline, ""):
                self.output_queue.put(line)
        finally:
            self.output_queue.put(None)

    def send_command(self, command, timeout=1800):
        if self.process.poll() is not None:
            raise RuntimeError(f"Joern 已退出，退出码: {self.process.returncode}")

        marker = f"__JOERN_CMD_DONE_{uuid.uuid4().hex}__"
        self.process.stdin.write(f"{command}\n")
        self.process.stdin.write(f'println("{marker}")\n')
        self.process.stdin.flush()

        output_lines = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Joern 命令执行超过 {timeout} 秒: {command}")
            try:
                line = self.output_queue.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"Joern 命令执行超过 {timeout} 秒: {command}")
            if line is None:
                break
            if marker in line:
                break
            output_lines.append(line)

        output = "".join(output_lines)
        if output.strip():
            print(output, end="" if output.endswith("\n") else "\n")
        return output

    def close(self):
        if self.process.poll() is None:
            try:
                self.send_command("exit", timeout=15)
            except (RuntimeError, TimeoutError):
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


def import_with_joern(repo_dir, project_name):
    project_dir = JOERN_WORKSPACE / project_name
    if project_dir.exists():
        print(f"Joern project already exists, skip import: {project_name}")
        return

    input_path = str(Path(repo_dir).resolve()).replace("\\", "\\\\").replace('"', '\\"')
    joern = JoernShell(JOERN_EXE)
    try:
        output = joern.send_command(
            f'importCode(inputPath="{input_path}", projectName="{project_name}")'
        )
        if not project_dir.exists():
            raise RuntimeError(
                f"Joern import finished but project was not created: {project_dir}\n{output}"
            )
    finally:
        joern.close()


def _is_git_repository(repo_dir):
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--git-dir"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def ensure_repo(lib_gitname, repo_url=None):
    repo_dir = DL_LIB_DIR / lib_gitname
    if repo_dir.exists():
        if not _is_git_repository(repo_dir):
            raise RuntimeError(f"目标目录不是 Git 仓库: {repo_dir}")
        return repo_dir

    if not repo_url:
        raise FileNotFoundError(
            f"仓库不存在: {repo_dir}。请通过 --repo-url 提供该第三方库的仓库地址。"
        )
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, str(repo_dir)])
    return repo_dir


def verify_commit(repo_dir, commit_hash):
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit_hash}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise ValueError(f"仓库中不存在 commit: {commit_hash}")


@contextmanager
def managed_worktree(repo_dir, commit_hash, label, keep=False):
    worktree_root = RUNTIME_DIR / "worktrees"
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree_path = worktree_root / f"{label}-{uuid.uuid4().hex[:8]}"

    run(
        [
            "git",
            "-C",
            str(repo_dir),
            "worktree",
            "add",
            "--detach",
            str(worktree_path),
            commit_hash,
        ]
    )
    try:
        yield worktree_path
    finally:
        if not keep:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_dir),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if worktree_path.exists():
                shutil.rmtree(worktree_path, ignore_errors=True)
            subprocess.run(
                ["git", "-C", str(repo_dir), "worktree", "prune"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            print(f"保留 worktree: {worktree_path}")


def _unlink_patterns(base_dir, patterns):
    if not base_dir.exists():
        return
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
                print(f"Deleted {path}")


def cleanup_intermediate_files(lib_name):
    documentation = ROOT / "documentation"
    for subdir, patterns in INTERMEDIATE_PATTERNS.items():
        formatted = [pattern.format(lib=lib_name) for pattern in patterns]
        _unlink_patterns(documentation / subdir, formatted)


def cleanup_transient_results(lib_name):
    patterns = [pattern.format(lib=lib_name) for pattern in RESULT_PATTERNS]
    _unlink_patterns(RESULTS_DIR, patterns)


def summarize_baseline(baseline_paths):
    summary = {
        "apis": 0,
        "cases": 0,
        "framework_load_failures": 0,
        "harness_failures": 0,
        "executed_cases": 0,
        "statuses": {},
    }
    seen_apis = set()
    for baseline_path in baseline_paths:
        try:
            data = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"无法读取 V1 baseline: {baseline_path}: {error}")
        if not isinstance(data, dict):
            raise RuntimeError(f"V1 baseline 顶层不是对象: {baseline_path}")
        for api_name, cases in data.items():
            seen_apis.add(api_name)
            if not isinstance(cases, list):
                continue
            for case in cases:
                if not isinstance(case, dict):
                    continue
                summary["cases"] += 1
                status = str(case.get("函数运行状态", "unknown"))
                summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
                result = str(case.get("函数返回结果", ""))
                if status == "error" and "run_api 函数加载失败" in result:
                    summary["framework_load_failures"] += 1
                if status == "harness_error":
                    summary["harness_failures"] += 1
                if status in {
                    "success",
                    "error",
                    "timeout",
                    "recursion_bug",
                    "library_bug",
                }:
                    summary["executed_cases"] += 1
    summary["apis"] = len(seen_apis)
    return summary


def move_results(lib_name, bug_id, record, metadata):
    target_dir = RESULTS_DIR / lib_name / f"{lib_name}_{bug_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    baseline_sources = sorted(RESULTS_DIR.glob(f"{lib_name}_v1_baseline*.json"))
    if not baseline_sources:
        raise RuntimeError(f"记录 {bug_id} 未生成 V1 baseline，不能视为执行成功。")
    baseline_summary = summarize_baseline(baseline_sources)
    if baseline_summary["cases"] == 0:
        raise RuntimeError(f"记录 {bug_id} 的 V1 baseline 没有任何测试用例。")
    if (
        baseline_summary["framework_load_failures"]
        + baseline_summary["harness_failures"]
        == baseline_summary["cases"]
    ):
        raise RuntimeError(
            f"记录 {bug_id} 的全部 {baseline_summary['cases']} 个用例"
            "均为测试框架错误，拒绝标记成功。"
        )
    if baseline_summary["executed_cases"] == 0:
        raise RuntimeError(
            f"记录 {bug_id} 没有任何用例实际进入目标 API，拒绝标记成功。"
        )

    for pattern in RESULT_PATTERNS:
        for old_path in target_dir.glob(pattern.format(lib=lib_name)):
            if old_path.is_file():
                old_path.unlink()

    moved = []
    for pattern in RESULT_PATTERNS:
        formatted = pattern.format(lib=lib_name)
        for source in sorted(RESULTS_DIR.glob(formatted)):
            if not source.is_file():
                continue
            destination = target_dir / source.name
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
            moved.append(destination.name)
            print(f"Moved {source} -> {destination}")

    failure_path = target_dir / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    metadata = dict(metadata)
    metadata["baseline_summary"] = baseline_summary
    manifest = {
        "status": "success",
        "record": record,
        "metadata": metadata,
        "artifacts": moved,
    }
    (target_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_failure(lib_name, record, error):
    target_dir = RESULTS_DIR / lib_name / f"{lib_name}_{record['bug_id']}"
    target_dir.mkdir(parents=True, exist_ok=True)
    for pattern in RESULT_PATTERNS:
        for old_path in target_dir.glob(pattern.format(lib=lib_name)):
            if old_path.is_file():
                old_path.unlink()
    manifest_path = target_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    payload = {
        "status": "failed",
        "record": record,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    (target_dir / "failure.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def should_use_joern(mode, lib_name, apis):
    if mode == "always":
        return True
    if mode == "never":
        return False
    return lib_name == "torch" or any(api.split("(", 1)[0].startswith("torch.") for api in apis)


def process_record(record, repo_dir, options):
    bug_id = record["bug_id"]
    bug_hash = record["bug_version"]
    fix_hash = record["fix_version"]
    project_name = f"{options.lib_name}_{re.sub(r'[^A-Za-z0-9_.-]', '_', bug_hash)}"
    environment_metadata = bugsinpy_record_metadata(options.lib_gitname, record)
    required_python = environment_metadata["required_python"]
    run_dir = RUNTIME_DIR / "runs" / options.lib_name / str(bug_id)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    executor_results = run_dir / "results"
    normalized_requirements = run_dir / "environment" / "requirements.txt"
    requirements_metadata = prepare_requirements(
        environment_metadata["requirements"],
        normalized_requirements,
    )
    completed = False
    repair_rounds_used = 0

    print("\n" + "=" * 80)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("Strict test Python: %s" % required_python)
    cleanup_intermediate_files(options.lib_name)
    cleanup_transient_results(options.lib_name)

    try:
        write_api_defs(options.lib_name, record["bug_api"])
        print("\n--- create strict Bug/Fix test environments ---")
        bug_test_environment = StrictTestEnvironment(
            required_python,
            run_dir / "envs" / "bug",
            options.python,
            provider=options.test_env_provider,
            explicit_python=options.test_python,
            index_url=options.index_url,
        ).create()
        fix_test_environment = StrictTestEnvironment(
            required_python,
            run_dir / "envs" / "fix",
            options.python,
            provider=options.test_env_provider,
            explicit_python=options.test_python,
            index_url=options.index_url,
        ).create()
        bug_test_environment.install_requirements(
            normalized_requirements,
            requirements_metadata,
        )
        fix_test_environment.install_requirements(
            normalized_requirements,
            requirements_metadata,
        )

        with managed_worktree(
            repo_dir,
            bug_hash,
            f"{options.lib_name}-{bug_id}-bug",
            keep=options.keep_worktrees,
        ) as bug_repo:
            bug_env = make_runtime_env(
                bug_repo,
                options.lib_name,
                options.lib_gitname,
                project_name,
            )
            bug_env["MOMO_REQUIRED_PYTHON"] = required_python

            if should_use_joern(options.joern, options.lib_name, record["bug_api"]):
                print("\n--- Joern importCode ---")
                import_with_joern(bug_repo, project_name)
            else:
                print("\n--- Joern skipped (Python-only library) ---")

            print("\n--- stage_2_function.py (bug version) ---")
            run_python(
                "stage_2_function.py",
                cwd=MOMO_DIR,
                env=bug_env,
                python_executable=options.python,
            )

            print("\n--- main.py --phase algo (bug version) ---")
            run_python(
                "main.py",
                args=["--phase", "algo", "--k", str(options.k)],
                cwd=MOMO_DIR,
                env=bug_env,
                python_executable=options.python,
            )

            bundle_dir = prepare_test_bundle(
                run_dir,
                options.lib_name,
                record,
            )

            bug_test_environment.run_setup(
                environment_metadata["setup"],
                bug_repo,
                environment_metadata["pythonpath"],
            )
            bug_test_environment.install_target(bug_repo)
            bug_test_env = bug_test_environment.runtime_env(
                bug_repo,
                environment_metadata["pythonpath"],
            )
            attempts_path = executor_results / "v1_attempts.json"
            repair_status_path = run_dir / "repair_status.json"
            pending_cases = None
            for repair_round in range(1, options.max_repair_rounds + 1):
                repair_rounds_used = repair_round
                print(
                    "\n--- path test probe and LLM repair "
                    f"(round {repair_round}/{options.max_repair_rounds}) ---"
                )
                run(
                    [
                        str(bug_test_environment.python),
                        str(MOMO_DIR / "test_executor.py"),
                        "--mode",
                        "probe",
                        "--bundle",
                        str(bundle_dir),
                        "--results",
                        str(executor_results),
                        "--expected-python",
                        required_python,
                        "--timeout",
                        str(options.case_timeout),
                    ],
                    cwd=bug_repo,
                    env=bug_test_env,
                )
                run_python(
                    "test_case_refiner.py",
                    args=[
                        "--bundle",
                        str(bundle_dir),
                        "--attempts",
                        str(attempts_path),
                        "--round",
                        str(repair_round),
                        "--status",
                        str(repair_status_path),
                    ],
                    cwd=MOMO_DIR,
                    env=bug_env,
                    python_executable=options.python,
                )
                repair_status = json.loads(
                    repair_status_path.read_text(encoding="utf-8")
                )
                pending_cases = int(repair_status.get("pending", 0))
                total_cases = int(repair_status.get("total", 0))
                if total_cases == 0:
                    raise RuntimeError(
                        "algorithm did not generate any path test cases"
                    )
                print(
                    "Path cases validated: "
                    f"{total_cases - pending_cases}/{total_cases}"
                )
                if pending_cases == 0:
                    break

            if pending_cases:
                raise RuntimeError(
                    f"{pending_cases} path test cases remained invalid after "
                    f"{options.max_repair_rounds} repair rounds"
                )

            print("\n--- materialize validated V1 baseline (bug version) ---")
            run(
                [
                    str(bug_test_environment.python),
                    str(MOMO_DIR / "test_executor.py"),
                    "--mode",
                    "v1",
                    "--bundle",
                    str(bundle_dir),
                    "--results",
                    str(executor_results),
                    "--expected-python",
                    required_python,
                    "--k",
                    str(options.k),
                    "--timeout",
                    str(options.case_timeout),
                ],
                cwd=bug_repo,
                env=bug_test_env,
            )

        with managed_worktree(
            repo_dir,
            fix_hash,
            f"{options.lib_name}-{bug_id}-fix",
            keep=options.keep_worktrees,
        ) as fix_repo:
            fix_test_environment.run_setup(
                environment_metadata["setup"],
                fix_repo,
                environment_metadata["pythonpath"],
            )
            fix_test_environment.install_target(fix_repo)
            fix_test_env = fix_test_environment.runtime_env(
                fix_repo,
                environment_metadata["pythonpath"],
            )
            print("\n--- test_executor.py --mode v2 (fix version) ---")
            run(
                [
                    str(fix_test_environment.python),
                    str(MOMO_DIR / "test_executor.py"),
                    "--mode",
                    "v2",
                    "--bundle",
                    str(bundle_dir),
                    "--results",
                    str(executor_results),
                    "--expected-python",
                    required_python,
                    "--timeout",
                    str(options.case_timeout),
                ],
                cwd=fix_repo,
                env=fix_test_env,
            )

        publish_executor_results(options.lib_name, executor_results)
        metadata = {
            "controller_python": str(BASE_PYTHON),
            "algorithm_python": str(options.python),
            "requested_python_version": required_python,
            "bug_test_environment": bug_test_environment.manifest(),
            "fix_test_environment": fix_test_environment.manifest(),
            "joern_mode": options.joern,
            "test_cases_per_path": options.k,
            "repair_rounds_used": repair_rounds_used,
            "max_repair_rounds": options.max_repair_rounds,
            "case_timeout": options.case_timeout,
            "test_bundle": str(bundle_dir),
        }
        move_results(options.lib_name, bug_id, record, metadata)
        completed = True
        return True
    finally:
        cleanup_intermediate_files(options.lib_name)
        cleanup_transient_results(options.lib_name)
        if completed and not options.keep_test_envs:
            shutil.rmtree(run_dir, ignore_errors=True)


def _validate_name(value, option_name):
    if not re.match(r"^[A-Za-z0-9_.-]+$", value):
        raise ValueError(f"{option_name} 包含非法字符: {value}")


def validate_local_layout(options, lib_file, joern_required):
    required_paths = [DATABASE_DIR, MOMO_DIR, DL_LIB_DIR, lib_file, options.python]
    if joern_required:
        required_paths.append(JOERN_EXE)
    missing = [Path(path) for path in required_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "当前项目布局缺少必要路径：\n" + "\n".join(str(path) for path in missing)
        )
    if shutil.which("git") is None:
        raise FileNotFoundError("未找到 git 可执行文件。")
    if options.k <= 0:
        raise ValueError("--k 必须大于 0。")
    if options.max_repair_rounds <= 0:
        raise ValueError("--max-repair-rounds 必须大于 0。")
    _validate_name(options.lib_name, "--lib-name")
    _validate_name(options.lib_gitname, "--lib-gitname")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run source-guided differential tests for a Python third-party library."
    )
    parser.add_argument("--lib-file", default=LIB_FILE)
    parser.add_argument("--lib-name", default=LIB_NAME)
    parser.add_argument("--lib-gitname", default=LIB_GITNAME)
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--start", type=int, default=START)
    parser.add_argument("--end", type=int, default=END)
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of complete test cases generated for each static path.",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=DEFAULT_MAX_REPAIR_ROUNDS,
        help="Maximum V1 execute-review-repair rounds before the record fails.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=BASE_PYTHON,
        help="Python interpreter for the algorithm environment only.",
    )
    parser.add_argument(
        "--test-env-provider",
        choices=["auto", "local", "uv", "pyenv"],
        default="auto",
        help="Provider used to create exact-version Bug/Fix test environments.",
    )
    parser.add_argument(
        "--test-python",
        type=Path,
        default=None,
        help="Optional exact-version local Python used by the local provider.",
    )
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--case-timeout", type=float, default=5.0)
    parser.add_argument("--keep-test-envs", action="store_true")
    parser.add_argument(
        "--joern",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto only enables Joern for currently supported native-code libraries.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-worktrees", action="store_true")
    parser.add_argument(
        "--no-bootstrap-runner",
        action="store_false",
        dest="bootstrap_runner",
        default=True,
        help="Do not create an isolated runner venv when core dependencies are missing.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate input, repository and commits without running the pipeline.",
    )
    return parser


def main(argv=None):
    options = build_parser().parse_args(argv)
    options.python = Path(
        os.path.abspath(str(options.python.expanduser()))
    )
    if options.test_python is not None:
        options.test_python = Path(
            os.path.abspath(str(options.test_python.expanduser()))
        )

    lib_file = Path(options.lib_file)
    if not lib_file.is_absolute():
        lib_file = DATABASE_DIR / lib_file

    records = parse_lib_file(lib_file)
    selected = records[options.start : options.end]
    joern_required = any(
        should_use_joern(options.joern, options.lib_name, record["bug_api"])
        for record in selected
    )
    validate_local_layout(options, lib_file, joern_required)
    repo_dir = ensure_repo(options.lib_gitname, options.repo_url)
    options.python = ensure_runner_python(
        options.python, bootstrap=options.bootstrap_runner
    )

    for record in selected:
        verify_commit(repo_dir, record["bug_version"])
        verify_commit(repo_dir, record["fix_version"])
        bugsinpy_record_metadata(options.lib_gitname, record)

    print(f"Loaded {len(records)} records from {lib_file}")
    print(f"Selected {len(selected)} records [{options.start}:{options.end}]")
    print(json.dumps(selected, ensure_ascii=False, indent=2))

    if options.preflight_only:
        print("Preflight passed.")
        return 0

    failures = []
    for index, record in enumerate(selected, start=options.start):
        print(f"\nProcessing record #{index} (bug_id={record['bug_id']})")
        try:
            process_record(record, repo_dir, options)
        except Exception as error:
            failures.append(record["bug_id"])
            write_failure(options.lib_name, record, error)
            print(
                f"记录 {record['bug_id']} 执行失败: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            if options.fail_fast:
                raise

    print(f"\n{'=' * 80}")
    print(f"Done. Processed: {len(selected)}, failed: {len(failures)}")
    if failures:
        print(f"Failed bug IDs: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    if os.name == "nt" and not is_admin():
        relaunch_as_admin()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("用户中断执行。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"启动失败: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
