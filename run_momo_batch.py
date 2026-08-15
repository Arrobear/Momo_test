import ctypes
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = ROOT / "documentation" / "database"
RESULTS_DIR = ROOT / "documentation" / "results"
MOMO_DIR = ROOT / "Momo_test"
CONFIG_PATH = MOMO_DIR / "config.py"
DL_LIB_DIR = ROOT / "documentation" / "dl_lib"
JOERN_EXE = ROOT / "joern-cli" / ("joern.bat" if os.name == "nt" else "joern")
if not JOERN_EXE.exists():
    JOERN_EXE = ROOT / "joern-cli" / "bin" / ("joern-cli.bat" if os.name == "nt" else "joern-cli")
JOERN_WORKSPACE = MOMO_DIR / "workspace"
BUGSINPY_DIR = DATABASE_DIR / "BugsInPy" / "projects"
SKIPPED_DIR = DATABASE_DIR / "skipped_entries"
RUNTIME_DIR = ROOT / ".momo_runtime"

LIB_FILE = "ansible.txt"
LIB_GITNAME = "ansible"
LIB_NAME = "ansible"
START = 0
END = None  # None 表示处理到最后一条记录

BASE_PYTHON = Path(sys.executable).resolve()

# macOS/base 环境运行策略：默认不创建 conda 环境、不安装依赖、不 pip install 待测库。
# 被测库通过 PYTHONPATH 指向当前项目下的 documentation/dl_lib/{lib}。
INSTALL_DEPENDENCIES = False
INSTALL_TARGET_PACKAGE = False


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
    result = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, args, str(MOMO_DIR), 1)
    if result <= 32:
        raise RuntimeError(f"管理员权限启动失败，ShellExecuteW 返回值：{result}")
    print("已请求管理员权限重新启动脚本，当前非管理员进程退出。")
    sys.exit(0)


def run(command, cwd=None, env=None):
    print(f"\n>>> {command}", flush=True)
    subprocess.run(command, cwd=cwd, shell=True, check=True, env=env)


def run_no_check(command, cwd=None, env=None):
    print(f"\n>>> {command}", flush=True)
    result = subprocess.run(command, cwd=cwd, shell=True, env=env)
    return result.returncode == 0


def quote_path(path):
    return shlex.quote(str(path))


def run_python(script, args="", cwd=None, env=None):
    script_path = quote_path(script)
    command = f"{quote_path(BASE_PYTHON)} -u {script_path}"
    if args:
        command = f"{command} {args}"
    run(command, cwd=cwd, env=env)


def make_base_env():
    cache_dir = RUNTIME_DIR / "cache"
    config_dir = RUNTIME_DIR / "config"
    joern_home = RUNTIME_DIR / "joern"
    for path in (RUNTIME_DIR, cache_dir, config_dir, joern_home):
        path.mkdir(parents=True, exist_ok=True)

    return {
        **os.environ,
        "MOMO_ROOT": str(ROOT),
        "JOERN_PATH": str(JOERN_EXE),
        "HOME": str(RUNTIME_DIR),
        "XDG_CACHE_HOME": str(cache_dir),
        "XDG_CONFIG_HOME": str(config_dir),
        "JOERN_HOME": str(joern_home),
    }


def make_runtime_env(repo_dir):
    python_paths = [
        str(repo_dir),
        str(repo_dir / "lib"),
        str(MOMO_DIR),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        python_paths.append(existing)

    env = make_base_env()
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def parse_lib_file(path):
    text = path.read_text(encoding="utf-8")
    records = []

    for block in re.split(r"^--------\s*$", text, flags=re.MULTILINE):
        lines = [line.rstrip() for line in block.splitlines() if line.strip() or (line.endswith("") and not line.strip())]
        if not lines:
            continue

        bug_id = None
        python_version = None
        bug_hash = None
        fix_hash = None
        apis = []
        in_api_section = False

        for line in lines:
            if not line.strip():
                in_api_section = False
                continue

            m = re.match(r"^bug_id:\s*(.+)$", line)
            if m:
                bug_id = int(m.group(1).strip())
                continue

            m = re.match(r"^python_version:\s*(.+)$", line)
            if m:
                python_version = m.group(1).strip()
                continue

            m = re.match(r"^buggy:\s*(.+)$", line)
            if m:
                bug_hash = m.group(1).strip()
                continue

            m = re.match(r"^fixed:\s*(.+)$", line)
            if m:
                fix_hash = m.group(1).strip()
                continue

            m = re.match(r"^bug_api:\s*$", line)
            if m:
                in_api_section = True
                continue

            if in_api_section:
                api_line = line.strip()
                if api_line and api_line != "(none)":
                    apis.append(api_line)

        if bug_id is None or not bug_hash or not fix_hash:
            print(f"警告：跳过缺少必要字段的条目 (bug_id={bug_id}, buggy={bug_hash}, fixed={fix_hash})")
            continue

        records.append({
            "bug_id": bug_id,
            "python_version": python_version or "",
            "bug_version": bug_hash,
            "fix_version": fix_hash,
            "bug_api": apis,
        })

    return records


def pip_install_env(env_name, args, cwd=None):
    """在当前 base Python 中执行 pip install。默认主流程不会调用此函数。"""
    run(f"{quote_path(BASE_PYTHON)} -m pip install {args}", cwd=cwd)


def replace_config_string(name, value):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(rf'^(\s*{re.escape(name)}\s*=\s*)(["\']).*?\2(.*)$', re.MULTILINE)
    new_text, count = pattern.subn(rf'\1"{value}"\3', text, count=1)
    if count != 1:
        raise ValueError(f"未在 {CONFIG_PATH} 中找到字符串配置项：{name}")
    CONFIG_PATH.write_text(new_text, encoding="utf-8")


def replace_config_bool(name, value):
    """替换 config.py 中的布尔/None 配置项（如 USE_SOURCE_RESOLVER = False）"""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(rf'^(\s*{re.escape(name)}\s*=\s*)(True|False|None)(.*)$', re.MULTILINE)
    new_text, count = pattern.subn(rf'\1{value}\3', text, count=1)
    if count != 1:
        raise ValueError(f"未在 {CONFIG_PATH} 中找到布尔配置项：{name}")
    CONFIG_PATH.write_text(new_text, encoding="utf-8")


def update_config(lib_name, lib_gitname, bug_id):
    joern_project = f"{lib_name}_{bug_id}"
    replace_config_string("lib_name", lib_name)
    replace_config_string("lib_gitname", lib_gitname)
    replace_config_string("joern_project", joern_project)
    replace_config_bool("USE_SOURCE_RESOLVER", "True")


def _is_prerelease_version(line):
    """检查依赖行是否指定了预发布版本号（dev/alpha/beta/rc），PyPI 通常不提供这些版本。"""
    import re
    return bool(re.search(r'==[^;]*\.dev\d+', line.lower())
                or re.search(r'==[^;]*(?:alpha|beta|rc)\d*', line.lower(), re.IGNORECASE))


def install_requirements(env_name, bug_id, lib_gitname):
    req_path = BUGSINPY_DIR / lib_gitname / "bugs" / str(bug_id) / "requirements.txt"
    if not req_path.exists():
        print(f"未找到 requirements.txt: {req_path}，跳过依赖安装。")
        return

    if not INSTALL_DEPENDENCIES:
        print(f"依赖安装已禁用，跳过: {req_path}")
        return

    print(f"安装依赖: {req_path}")

    # 过滤掉 -e git+ 可编辑安装行（它们指向的 commit 可能已不在当前仓库中）
    # 以及已知的与 setuptools 不兼容的老旧包
    KNOWN_BROKEN = {"accessify", "funcsigs", "pathlib2", "scandir", "functools32", "typing"}
    lines = req_path.read_bytes()

    # 处理 UTF-16 编码（用 UTF-8 读取会看到空字节，decode 会失败或给出一堆 NUL）
    try:
        text = lines.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = lines.decode("utf-16-le")

    filtered = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            filtered.append(line)
        elif stripped.startswith("-e"):
            print(f"跳过可编辑安装: {stripped}")
        else:
            # 检查是否为已知问题包
            pkg_name = stripped.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0].strip()
            if pkg_name.lower() in {k.lower() for k in KNOWN_BROKEN}:
                print(f"跳过已知不兼容包: {stripped}")
            # 跳过 dev/alpha/beta/rc 预发布版本（PyPI 通常不提供这些版本）
            elif _is_prerelease_version(stripped):
                print(f"跳过预发布版本: {stripped}")
            else:
                filtered.append(line)

    filtered_text = "\n".join(filtered)
    if filtered_text.strip():
        temp_req = req_path.with_suffix(".tmp.txt")
        temp_req.write_text(filtered_text, encoding="utf-8")
        try:
            pip_install_env(env_name, f'-r "{temp_req}"')
        except subprocess.CalledProcessError:
            print("警告：部分依赖安装失败，尝试继续...")
        finally:
            temp_req.unlink(missing_ok=True)
    else:
        print("过滤后无有效依赖，跳过安装。")


def write_api_defs(lib_name, apis):
    api_dir = ROOT / "documentation" / "lib_api"
    api_dir.mkdir(parents=True, exist_ok=True)
    api_path = api_dir / f"{lib_name}_APIdef.txt"
    api_path.write_text("\n".join(apis) + "\n", encoding="utf-8")
    print(f"Wrote {api_path}")


class JoernShell:
    def __init__(self, joern_bat_path):
        self.process = subprocess.Popen(
            [str(joern_bat_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="gbk" if os.name == "nt" else "utf-8",
            errors="replace",
            shell=False,
            cwd=MOMO_DIR,
            env=make_base_env(),
        )

    def send_command(self, cmd):
        marker = f"__JOERN_CMD_DONE_{uuid.uuid4().hex}__"
        self.process.stdin.write(f"{cmd}\n")
        self.process.stdin.flush()
        self.process.stdin.write(f'println("{marker}")\n')
        self.process.stdin.flush()

        output_lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
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
                self.send_command("exit")
            finally:
                self.process.terminate()


def import_with_joern(lib_gitname, lib_name, commit_hash):
    project_name = f"{lib_name}_{commit_hash}"
    project_dir = JOERN_WORKSPACE / project_name
    if project_dir.exists():
        print(f"Joern project already exists, skip import: {project_name}")
        return

    input_path = str(DL_LIB_DIR / lib_gitname).replace("\\", "\\\\").replace('"', '\\"')
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


def ensure_repo(lib_gitname):
    """确保 dl_lib 下存在对应仓库，不存在则尝试 clone。"""
    repo_dir = DL_LIB_DIR / lib_gitname
    if repo_dir.exists():
        return repo_dir
    print(f"仓库目录不存在: {repo_dir}")
    clone_url = f"https://github.com/psf/{lib_gitname}.git"
    print(f"尝试 clone: {clone_url}")
    run(f"git clone {clone_url} \"{repo_dir}\"")
    return repo_dir


def checkout_version(repo_dir, commit_hash):
    run(f"git checkout {commit_hash}", cwd=repo_dir)


def validate_local_layout():
    required_paths = [DATABASE_DIR, MOMO_DIR, DL_LIB_DIR, JOERN_EXE]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"当前项目布局缺少必要路径：\n{missing_text}")


def move_results(lib_name, bug_id):
    target_dir = RESULTS_DIR / lib_name / f"{lib_name}_{bug_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        f"{lib_name}_diff_report.json",
        f"{lib_name}_recursion_bugs.json",
        f"{lib_name}_timeout_bugs.json",
        f"{lib_name}_v1_baseline.json",
    ]:
        source = RESULTS_DIR / filename
        if source.exists():
            destination = target_dir / filename
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))
            print(f"Moved {source} -> {destination}")


def cleanup_intermediate_files(lib_name):
    paths = [
        ROOT / "documentation" / "api_guards" / f"{lib_name}_api_guards.json",
        ROOT / "documentation" / "api_input" / f"{lib_name}_default_inputs_0.json",
        ROOT / "documentation" / "api_input" / f"{lib_name}_inputs_0.json",
        ROOT / "documentation" / "api_src_code" / f"{lib_name}_api_sources.json",
        ROOT / "documentation" / "arg_boundary" / f"cut_{lib_name}_boundary_0.json",
        ROOT / "documentation" / "arg_combinations" / f"{lib_name}_cut_combinations_0.json",
        ROOT / "documentation" / "arg_combinations" / f"{lib_name}_combinations_0.json",
        ROOT / "documentation" / "arg_space" / f"{lib_name}_arg_space_0.json",
        ROOT / "documentation" / "error_combinations" / f"error_{lib_name}_combinations.json",
        ROOT / "documentation" / "conditions" / f"{lib_name}_conditions.json",
        ROOT / "documentation" / "test_cases" / f"{lib_name}_case_0.json",
    ]

    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Deleted {path}")


def process_record(record, lib_gitname, lib_name):
    bug_id = record["bug_id"]
    python_version = record["python_version"]
    bug_hash = record["bug_version"]
    fix_hash = record["fix_version"]
    repo_dir = ensure_repo(lib_gitname)
    env_name = "base"
    env_with_path = make_runtime_env(repo_dir)

    print("\n" + "=" * 80)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    # 13) 清理中间文件
    print("\n--- 清理中间文件 ---")
    cleanup_intermediate_files(lib_name)

    # 1) 环境准备：使用启动脚本的当前 Python，不切换 conda 环境
    print("\n--- 环境准备 (base/current Python) ---")
    print(f"Project root: {ROOT}")
    print(f"Python: {BASE_PYTHON}")
    print(f"Joern: {JOERN_EXE}")
    print(f"Target python_version metadata: {python_version or '(none)'}")

    # 2) 依赖准备：默认只记录并跳过，不向 base 环境安装包
    print("\n--- 依赖准备 ---")
    install_requirements(env_name, bug_id, lib_gitname)

    # 3) 更新源代码为 bug 版本
    print("\n--- 更新源代码为 bug 版本 ---")
    checkout_version(repo_dir, bug_hash)

    # 4) 修改 config.py
    print("\n--- 更新 config.py ---")
    update_config(lib_name, lib_gitname, bug_id)

    # 5) 待测库加载策略：默认不安装到 base 环境，只通过 PYTHONPATH 指向源码
    print("\n--- 待测库加载 (bug 版本) ---")
    if INSTALL_TARGET_PACKAGE:
        if not run_no_check(f"{quote_path(BASE_PYTHON)} -m pip install .", cwd=repo_dir, env=env_with_path):
            print("警告：待测库安装失败，尝试使用 pip install -e (开发模式)...")
            if not run_no_check(f"{quote_path(BASE_PYTHON)} -m pip install -e .", cwd=repo_dir, env=env_with_path):
                print("警告：开发模式安装也失败，将通过 PYTHONPATH 兜底。")
    else:
        print(f"跳过 pip install，使用 PYTHONPATH: {env_with_path['PYTHONPATH']}")

    # 6) 写入 API 定义文件
    print("\n--- 写入 API 定义文件 ---")
    write_api_defs(lib_name, record["bug_api"])

    # 7) 启动 Joern 并 importCode
    print("\n--- Joern importCode ---")
    import_with_joern(lib_gitname, lib_name, bug_hash)

    # 8) 执行 stage_2_function.py → 当前 base Python
    print("\n--- stage_2_function.py (base/current Python) ---")
    run_python("stage_2_function.py", cwd=MOMO_DIR, env=env_with_path)

    # 9a) 执行 main.py 算法阶段 → 当前 base Python
    print("\n--- main.py --phase algo (base/current Python) ---")
    run_python("main.py", "--phase algo", cwd=MOMO_DIR, env=env_with_path)

    # 9b) 执行 main.py 测试阶段 → 当前 base Python（bug 版本测试，生成 V1 基线）
    print("\n--- main.py --phase test (base/current Python) ---")
    run_python("main.py", "--phase test --k 500", cwd=MOMO_DIR, env=env_with_path)

    # 10) 切换到 fix 版本，仍只通过 PYTHONPATH 加载源码
    print("\n--- 切换到 fix 版本 ---")
    checkout_version(repo_dir, fix_hash)
    if INSTALL_TARGET_PACKAGE:
        if not run_no_check(f"{quote_path(BASE_PYTHON)} -m pip install .", cwd=repo_dir, env=env_with_path):
            print("警告：fix 版本安装失败。")
        if not run_no_check(f"{quote_path(BASE_PYTHON)} -m pip install -e .", cwd=repo_dir, env=env_with_path):
            print("警告：fix 版本开发模式安装也失败，将通过 PYTHONPATH 兜底。")
    else:
        print("跳过 fix 版本 pip install，继续使用 PYTHONPATH 加载源码。")

    # 11) 执行 entrance.py → 当前 base Python（fix 版本测试，生成 V2 + diff）
    print("\n--- entrance.py (base/current Python) ---")
    run_python("entrance.py", cwd=MOMO_DIR, env=env_with_path)

    # 12) 移动结果文件
    print("\n--- 移动结果文件 ---")
    move_results(lib_name, bug_id)

    # 13) 清理中间文件
    print("\n--- 清理中间文件 ---")
    cleanup_intermediate_files(lib_name)

    return True


def main():
    validate_local_layout()

    lib_file = Path(LIB_FILE)
    if not lib_file.is_absolute():
        lib_file = DATABASE_DIR / lib_file

    records = parse_lib_file(lib_file)
    selected = records[START:END]

    print(f"Loaded {len(records)} records from {lib_file}")
    print(json.dumps(selected, ensure_ascii=False, indent=2))

    for index, record in enumerate(selected, start=START):
        print(f"\nProcessing record #{index} (bug_id={record['bug_id']})")
        process_record(record, LIB_GITNAME, LIB_NAME)

    # 恢复 config.py 默认状态
    replace_config_bool("USE_SOURCE_RESOLVER", "False")

    print(f"\n{'=' * 80}")
    print(f"Done. Processed: {len(selected)} records.")


if __name__ == "__main__":
    if os.name == "nt" and not is_admin():
        relaunch_as_admin()

    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败，退出码：{exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
