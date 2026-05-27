import ctypes
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(r"C:\Users\86184\Desktop\Papers")
RESULTS_DIR = ROOT / "documentation" / "results"
MOMO_DIR = ROOT / "Momo_test"
CONFIG_PATH = MOMO_DIR / "config.py"
DL_LIB_DIR = ROOT / "dl_lib"
JOERN_BAT = Path(r"C:\Users\86184\Desktop\joern-cli\joern.bat")
JOERN_WORKSPACE = MOMO_DIR / "workspace"
CONDA_ENV = "momo_test"

from config import lib_gitname, lib_name, test_version

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


def replace_config_string(name, value):
    text = CONFIG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(CONFIG_ASSIGN_RE_TEMPLATE.format(name=re.escape(name)), re.MULTILINE)
    new_text, count = pattern.subn(rf'\1"{value}"\3', text, count=1)
    if count != 1:
        raise ValueError(f"未在 {CONFIG_PATH} 中找到字符串配置项：{name}")
    CONFIG_PATH.write_text(new_text, encoding="utf-8")


def update_config_joern_project(project_name):
    replace_config_string("joern_project", project_name)


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


def move_results(lib_name):
    target_dir = RESULTS_DIR / lib_name
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


def main():
    target_commit = test_version[0]
    baseline_commit = test_version[1]
    repo_dir = DL_LIB_DIR / lib_gitname
    project_name = f"{lib_name}_{target_commit}"

    print("=" * 80)
    print(f"Library: {lib_name}")
    print(f"Git repo: {lib_gitname}")
    print(f"Target version: {target_commit}")
    print(f"Baseline version: {baseline_commit}")
    print("=" * 80)

    # Step 1: Install the target library version
    print("\n[Step 1] Installing target library version...")
    install_local_lib(repo_dir)

    # Step 2: Import code into Joern
    print("\n[Step 2] Importing code into Joern...")
    import_with_joern(lib_gitname, lib_name, target_commit)

    # Step 3: Update config joern_project
    print("\n[Step 3] Updating config joern_project...")
    update_config_joern_project(project_name)

    # Step 4: Run extract_func_name.py
    print("\n[Step 4] Running extract_func_name.py...")
    run_conda("python -u extract_func_name.py", cwd=MOMO_DIR)

    # Step 5: Run stage_2_function.py
    print("\n[Step 5] Running stage_2_function.py...")
    run_conda("python -u stage_2_function.py", cwd=MOMO_DIR)

    # Step 6: Run main.py
    print("\n[Step 6] Running main.py...")
    run_conda("python -u main.py", cwd=MOMO_DIR)

    # Step 7: Checkout baseline version
    print("\n[Step 7] Checking out baseline version...")
    checkout_version(repo_dir, baseline_commit)

    # Step 8: Run entrance.py
    print("\n[Step 8] Running entrance.py...")
    run_conda("python -u entrance.py", cwd=MOMO_DIR)

    # Step 9: Move results
    print("\n[Step 9] Moving results...")
    move_results(lib_name)

    print("\n" + "=" * 80)
    print("All steps completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    if not is_admin():
        relaunch_as_admin()

    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"命令执行失败，退出码：{exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
