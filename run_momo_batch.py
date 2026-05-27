import ctypes
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(r"C:\Users\86184\Desktop\Papers")
FIXES_DIR = ROOT / "documentation" / "database" / "llm_verified_fixes"
RESULTS_DIR = ROOT / "documentation" / "results"
MOMO_DIR = ROOT / "Momo_test"
CONFIG_PATH = MOMO_DIR / "config.py"
DL_LIB_DIR = ROOT / "dl_lib"
JOERN_BAT = Path(r"C:\Users\86184\Desktop\joern-cli\joern.bat")
JOERN_WORKSPACE = MOMO_DIR / "workspace"
CONDA_ENV = "momo_test"

LIB_FILE = "boolean.py_summary.txt"
LIB_GITNAME = "boolean.py_source"
LIB_NAME = "boolean"
# START从0开始
START = 0
# END = None 表示处理到最后一组；设为具体数字表示处理到该组（不含）
END = None

CLEAN = True

BUG_RE = re.compile(r"^Bug version:\s*(.*?)\s*\(([0-9a-fA-F]+)\)\s*$")
FIX_RE = re.compile(r"^Fix version:\s*(.*?)\s*\(([0-9a-fA-F]+)\)\s*$")
CONFIG_ASSIGN_RE_TEMPLATE = r'^(\s*{name}\s*=\s*)(["\']).*?\2(.*)$'


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


def run_conda(command, cwd=None):
    run(f'conda run --no-capture-output -n {CONDA_ENV} {command}', cwd=cwd)


def parse_fix_file(path):
    text = path.read_text(encoding="utf-8")
    records = []

    for block in re.split(r"^---\s*$", text, flags=re.MULTILINE):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        bug_hash = None
        fix_hash = None
        apis = []

        for line in lines:
            bug_match = BUG_RE.match(line)
            fix_match = FIX_RE.match(line)
            if bug_match:
                bug_hash = bug_match.group(2)
            elif fix_match:
                fix_hash = fix_match.group(2)
            elif not line.startswith("SIG_CHANGED:"):
                apis.append(line)

        if not bug_hash or not fix_hash:
            raise ValueError(f"无法解析数据块：\n{block}")

        records.append({
            "bug_version": bug_hash,
            "fix_version": fix_hash,
            "bug_api": apis,
        })

    return records


def replace_config_string(name, value):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(CONFIG_ASSIGN_RE_TEMPLATE.format(name=re.escape(name)), re.MULTILINE)
    new_text, count = pattern.subn(rf'\1"{value}"\3', text, count=1)
    if count != 1:
        raise ValueError(f"未在 {CONFIG_PATH} 中找到字符串配置项：{name}")
    CONFIG_PATH.write_text(new_text, encoding="utf-8")


def update_config(lib_name, commit_hash):
    joern_project = f"{lib_name}_{commit_hash}"
    replace_config_string("lib_name", lib_name)
    replace_config_string("conmmit_hash", commit_hash)
    replace_config_string("joern_project", joern_project)


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
        output = joern.send_command(f'importCode(inputPath="{input_path}", projectName="{project_name}")')
        if not project_dir.exists():
            raise RuntimeError(f"Joern import finished but project was not created: {project_dir}\n{output}")
    finally:
        joern.close()



def checkout_version(repo_dir, commit_hash):
    run(f"git checkout {commit_hash}", cwd=repo_dir)


def install_local_lib(repo_dir):
    run_conda("python -m pip install --upgrade --force-reinstall .", cwd=repo_dir)


def cleanup_joern_import_script(lib_name, commit_hash):
    script_path = MOMO_DIR / f"joern_import_{lib_name}_{commit_hash}.sc"
    if script_path.exists():
        script_path.unlink()
        print(f"Deleted {script_path}")


def cleanup_intermediate_files(lib_name):
    paths = [
        ROOT / "documentation" / "api_guards" / f"{lib_name}_api_guards.json",
        ROOT / "documentation" / "api_input" / f"{lib_name}_default_inputs_0.json",
        ROOT / "documentation" / "api_input" / f"{lib_name}_inputs_0.json",
        ROOT / "documentation" / "api_src_code" / f"{lib_name}_api_sources.json",
        ROOT / "documentation" / "arg_boundary" / f"cut_{lib_name}_boundary_0.json",
        ROOT / "documentation" / "arg_combinations" / f"{lib_name}_cut_combinations_0.json",
        ROOT / "documentation" / "arg_space" / f"{lib_name}_arg_space_0.json",
        ROOT / "documentation" / "error_combinations" / f"error_{lib_name}_combinations.json",
        ROOT / "documentation" / "conditions" / f"{lib_name}_conditions.json",
        ROOT / "documentation" / "test_cases" / f"{lib_name}_case_0.json",
    ]

    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Deleted {path}")


def move_results(lib_name, bug_hash):
    target_dir = RESULTS_DIR / f"{lib_name}_{bug_hash}"
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
    

def process_record(record, lib_gitname, lib_name):
    bug_hash = record["bug_version"]
    fix_hash = record["fix_version"]
    repo_dir = DL_LIB_DIR / lib_gitname

    print("\n" + "=" * 80)
    print(json.dumps(record, ensure_ascii=False, indent=2))

    checkout_version(repo_dir, bug_hash)
    update_config(lib_name, bug_hash)
    write_api_defs(lib_name, record["bug_api"])
    install_local_lib(repo_dir)
    import_with_joern(lib_gitname, lib_name, bug_hash)
    run_conda("python -u stage_2_function.py", cwd=MOMO_DIR)
    run_conda("python -u main.py", cwd=MOMO_DIR)
    checkout_version(repo_dir, fix_hash)
    install_local_lib(repo_dir)
    run_conda("python -u entrance.py", cwd=MOMO_DIR)
    move_results(lib_name, bug_hash)
    cleanup_intermediate_files(lib_name)
    cleanup_joern_import_script(lib_name, bug_hash)


def main():
    if CLEAN:
        cleanup_intermediate_files(LIB_NAME)

    else:
        lib_file = Path(LIB_FILE)
        if not lib_file.is_absolute():
            lib_file = FIXES_DIR / lib_file

        records = parse_fix_file(lib_file)
        selected = records[START:END]

        print(f"Loaded {len(records)} records from {lib_file}")
        print(json.dumps(selected, ensure_ascii=False, indent=2))

        for index, record in enumerate(selected, start=START):
            print(f"\nProcessing record #{index}")
            process_record(record, LIB_GITNAME, LIB_NAME)


if __name__ == "__main__":
    if not is_admin():
        relaunch_as_admin()

    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败，退出码：{exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
