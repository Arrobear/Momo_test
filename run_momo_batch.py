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
DEFAULT_K = 10
BASE_PYTHON = Path(sys.executable).resolve()

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
        return Path(python_executable).resolve()
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
    return worker_python.resolve()


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


def move_results(lib_name, bug_id, record, metadata):
    target_dir = RESULTS_DIR / lib_name / f"{lib_name}_{bug_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

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

    baseline_prefix = f"{lib_name}_v1_baseline"
    if not any(name.startswith(baseline_prefix) for name in moved):
        raise RuntimeError(f"记录 {bug_id} 未生成 V1 baseline，不能视为执行成功。")

    failure_path = target_dir / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
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

    print("\n" + "=" * 80)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    cleanup_intermediate_files(options.lib_name)
    cleanup_transient_results(options.lib_name)

    if options.install_dependencies:
        install_requirements(bug_id, options.lib_gitname, options.python)

    try:
        write_api_defs(options.lib_name, record["bug_api"])

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
            if options.install_target_package:
                install_target_package(bug_repo, bug_env, options.python)

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
                args=["--phase", "algo"],
                cwd=MOMO_DIR,
                env=bug_env,
                python_executable=options.python,
            )

            print("\n--- main.py --phase test --test-mode v1 (bug version) ---")
            run_python(
                "main.py",
                args=[
                    "--phase",
                    "test",
                    "--test-mode",
                    "v1",
                    "--k",
                    str(options.k),
                ],
                cwd=MOMO_DIR,
                env=bug_env,
                python_executable=options.python,
            )

        with managed_worktree(
            repo_dir,
            fix_hash,
            f"{options.lib_name}-{bug_id}-fix",
            keep=options.keep_worktrees,
        ) as fix_repo:
            fix_env = make_runtime_env(
                fix_repo,
                options.lib_name,
                options.lib_gitname,
                project_name,
            )
            if options.install_target_package:
                install_target_package(fix_repo, fix_env, options.python)

            print("\n--- entrance.py --test-mode v2 (fix version) ---")
            run_python(
                "entrance.py",
                args=["--test-mode", "v2"],
                cwd=MOMO_DIR,
                env=fix_env,
                python_executable=options.python,
            )

        metadata = {
            "controller_python": str(BASE_PYTHON),
            "worker_python": str(options.python),
            "requested_python_version": record["python_version"],
            "joern_mode": options.joern,
            "test_cases_per_api": options.k,
        }
        move_results(options.lib_name, bug_id, record, metadata)
        return True
    finally:
        cleanup_intermediate_files(options.lib_name)
        cleanup_transient_results(options.lib_name)


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
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--python", type=Path, default=BASE_PYTHON)
    parser.add_argument(
        "--joern",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto only enables Joern for currently supported native-code libraries.",
    )
    parser.add_argument(
        "--install-dependencies",
        action="store_true",
        default=INSTALL_DEPENDENCIES,
    )
    parser.add_argument(
        "--install-target-package",
        action="store_true",
        default=INSTALL_TARGET_PACKAGE,
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
    options.python = options.python.expanduser().resolve()

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
