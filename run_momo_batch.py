import ctypes
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(r"C:\Users\86184\Desktop\Papers")
DATABASE_DIR = ROOT / "documentation" / "database"
RESULTS_DIR = ROOT / "documentation" / "results"
MOMO_DIR = ROOT / "Momo_test"
CONFIG_PATH = MOMO_DIR / "config.py"
DL_LIB_DIR = ROOT / "documentation" / "dl_lib"
JOERN_BAT = Path(r"C:\Users\86184\Desktop\joern-cli\joern.bat")
JOERN_WORKSPACE = MOMO_DIR / "workspace"
BUGSINPY_DIR = DATABASE_DIR / "BugsInPy" / "projects"
SKIPPED_DIR = DATABASE_DIR / "skipped_entries"

LIB_FILE = "black.txt"
LIB_GITNAME = "black"
LIB_NAME = "black"
START = 0
END = None

MOMO_ENV = "momo_test"  # 算法环境（有 ML 依赖）


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except OSError:
        return False


def relaunch_as_admin():
    script = Path(__file__).resolve()
    args = " ".join([f'"{script}"', *[f'"{arg}"' for arg in sys.argv[1:]]])
    result = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, args, str(MOMO_DIR), 1)
    if result <= 32:
        raise RuntimeError(f"管理员权限启动失败，ShellExecuteW 返回值：{result}")
    print("已请求管理员权限重新启动脚本，当前非管理员进程退出。")
    sys.exit(0)


def run(command, cwd=None):
    print(f"\n>>> {command}", flush=True)
    subprocess.run(command, cwd=cwd, shell=True, check=True)


def run_no_check(command, cwd=None):
    print(f"\n>>> {command}", flush=True)
    result = subprocess.run(command, cwd=cwd, shell=True)
    return result.returncode == 0


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


def conda_env_name(python_version):
    return f"momo_test_{python_version}"


def conda_env_exists(env_name):
    result = subprocess.run("conda env list", shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.strip() and not line.startswith("#"):
            parts = line.split()
            if parts and parts[0] == env_name:
                return True
    return False


def activate_conda_env(env_name):
    if conda_env_exists(env_name):
        print(f"conda 环境 {env_name} 已存在，直接激活。")
    else:
        python_version = env_name.replace("momo_test_", "")
        print(f"conda 环境 {env_name} 不存在，正在创建 (python={python_version}) ...")
        run(f"conda create -n {env_name} python={python_version} -y")
    run(f"conda activate {env_name}")
    run(f"conda run --no-capture-output -n {env_name} python -m pip install --upgrade setuptools wheel psutil json_repair pyyaml")


def pip_install_env(env_name, args, cwd=None):
    """在指定 conda 环境中执行 pip install。"""
    run(f'conda run --no-capture-output -n {env_name} pip install {args}', cwd=cwd)


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


def install_requirements(env_name, bug_id, lib_gitname):
    req_path = BUGSINPY_DIR / lib_gitname / "bugs" / str(bug_id) / "requirements.txt"
    if req_path.exists():
        print(f"安装依赖: {req_path}")
        pip_install_env(env_name, f'-r "{req_path}"')
    else:
        print(f"未找到 requirements.txt: {req_path}，跳过依赖安装。")


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
            encoding="gbk",
            errors="replace",
            shell=True,
            cwd=MOMO_DIR,
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

    input_path = str(DL_LIB_DIR / lib_gitname).replace("\\", "\\\\")
    joern = JoernShell(JOERN_BAT)
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
    env_name = conda_env_name(python_version)

    print("\n" + "=" * 80)
    print(json.dumps(record, ensure_ascii=False, indent=2))

    # 1) 环境准备：确保 momo_test（算法）和版本 env（测试）都存在
    print("\n--- 环境准备 ---")
    run("conda deactivate")
    activate_conda_env(MOMO_ENV)
    activate_conda_env(env_name)

    # 2) 依赖准备 → 版本 env
    print("\n--- 依赖准备 ---")
    install_requirements(env_name, bug_id, lib_gitname)

    # 3) 更新源代码为 bug 版本
    print("\n--- 更新源代码为 bug 版本 ---")
    checkout_version(repo_dir, bug_hash)

    # 4) 修改 config.py
    print("\n--- 更新 config.py ---")
    update_config(lib_name, lib_gitname, bug_id)

    # 5) 安装待测库 → 版本 env（测试用）
    print("\n--- 安装待测库 (bug 版本) ---")
    pip_install_env(env_name, ".", cwd=repo_dir)

    # 6) 写入 API 定义文件
    print("\n--- 写入 API 定义文件 ---")
    write_api_defs(lib_name, record["bug_api"])

    # 7) 启动 Joern 并 importCode
    print("\n--- Joern importCode ---")
    import_with_joern(lib_gitname, lib_name, bug_hash)

    # 8) 执行 stage_2_function.py → momo_test（算法）
    print("\n--- stage_2_function.py (momo_test) ---")
    run(f'conda run --no-capture-output -n {MOMO_ENV} python -u stage_2_function.py', cwd=MOMO_DIR)

    # 9a) 执行 main.py 算法阶段 → momo_test（算法）
    print("\n--- main.py --phase algo (momo_test) ---")
    run(f'conda run --no-capture-output -n {MOMO_ENV} python -u main.py --phase algo', cwd=MOMO_DIR)

    # 9b) 执行 main.py 测试阶段 → 版本 env（bug 版本测试，生成 V1 基线）
    print("\n--- main.py --phase test (版本 env) ---")
    run(f'conda run --no-capture-output -n {env_name} python -u main.py --phase test --k 500', cwd=MOMO_DIR)

    # 10) 切换到 fix 版本并安装 → 版本 env
    print("\n--- 切换到 fix 版本 ---")
    checkout_version(repo_dir, fix_hash)
    pip_install_env(env_name, ".", cwd=repo_dir)

    # 11) 执行 entrance.py → 版本 env（fix 版本测试，生成 V2 + diff）
    print("\n--- entrance.py (版本 env) ---")
    run(f'conda run --no-capture-output -n {env_name} python -u entrance.py', cwd=MOMO_DIR)

    # 12) 移动结果文件
    print("\n--- 移动结果文件 ---")
    move_results(lib_name, bug_id)

    # 13) 清理中间文件
    print("\n--- 清理中间文件 ---")
    cleanup_intermediate_files(lib_name)

    return True


def main():
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
    if not is_admin():
        relaunch_as_admin()

    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败，退出码：{exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
