"""Run one Python-library version pair in a fresh directory.

Use a disposable Python environment containing Momo's dependencies:
    python -m rq3.run --manifest runs.json --run demo --root /path/to/output

--repository-root contains existing Git repositories. --root contains rq3_runs;
an existing run directory is never reused. Each run keeps its own repository
clone, documentation, seed bundle and stage logs, including failed attempts.
The selected interpreter installs both library versions in sequence.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rq3.git import resolve_commit
from rq3.manifest import SOURCE_ROOT, default_repository_root, default_root, load_run_spec
from rq3.seed_loader import resolve_input_seed_bundle


DOCUMENTATION_DIRS = (
    "lib_api", "api_guards", "api_src_code", "arg_space", "conditions",
    "arg_combinations", "error_combinations", "arg_boundary", "api_input",
    "test_cases", "results",
)


def build_environment(run_spec, run_dir):
    environment = os.environ.copy()
    for name in ("MOMO_API_INCLUDE", "MOMO_INPUT_SEEDS", "MOMO_INPUT_SEEDS_FILE"):
        environment.pop(name, None)
    environment.update({
        "MOMO_ROOT_PATH": str(run_dir),
        "MOMO_REPOSITORY_ROOT": str(run_dir / "repositories"),
        "MOMO_LIB_NAME": run_spec.get("import_name", run_spec["library"]),
        "MOMO_LIB_GITNAME": run_spec["repo_dir"],
        "MOMO_REFERENCE_COMMIT": run_spec["resolved_reference"],
        "MOMO_CANDIDATE_COMMIT": run_spec["resolved_candidate"],
        "MOMO_INPUT_SEEDS_FILE": str(run_dir / "input_seeds.json"),
        "PYTHONPATH": str(SOURCE_ROOT),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    })
    if run_spec.get("api_include"):
        environment["MOMO_API_INCLUDE"] = json.dumps(run_spec["api_include"])
    return environment


def run_command(command, cwd, environment, log_path):
    print(f"Running {log_path.stem}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=cwd, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log_path}")


def run_one(run_spec, manifest_dir, root, repository_root, python=sys.executable, k=500):
    run_dir = (Path(root) / "rq3_runs" / run_spec["run_id"]).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}; use a new run_id or --root")
    source_repo = (Path(repository_root) / run_spec["repo_dir"]).resolve()
    resolved = dict(run_spec)
    resolved["resolved_reference"] = resolve_commit(source_repo, run_spec["reference"])
    resolved["resolved_candidate"] = resolve_commit(source_repo, run_spec["candidate"])
    # Validate artifact commit hashes before creating any output or installing a library.
    bundle = resolve_input_seed_bundle(resolved, manifest_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    work = run_dir / "work"
    work.mkdir()
    logs = run_dir / "logs"
    logs.mkdir()
    repositories = run_dir / "repositories"
    repositories.mkdir()
    for name in DOCUMENTATION_DIRS:
        (run_dir / "documentation" / name).mkdir(parents=True)
    (run_dir / "input_seeds.json").write_text(
        json.dumps(bundle["input_seeds"], ensure_ascii=True, allow_nan=False), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(
        json.dumps({**resolved, "seed_sources": bundle["sources"], "k": k}, indent=2), encoding="utf-8"
    )
    environment = build_environment(resolved, run_dir)
    repo = repositories / run_spec["repo_dir"]

    def execute(label, command, cwd=work):
        run_command(command, cwd, environment, logs / f"{label}.log")

    def stage(name):
        execute(name, [python, "-m", "rq3.stage", name, "--k", str(k)])

    execute("clone", ["git", "clone", "--no-hardlinks", "--no-checkout", str(source_repo), str(repo)])
    for phase in ("reference", "candidate"):
        if phase == "candidate":
            # Only this run's fresh clone is cleaned; source_repo is never modified.
            execute("clean_candidate", ["git", "clean", "-fdx"], repo)
        execute(f"checkout_{phase}", ["git", "checkout", "--detach", resolved[f"resolved_{phase}"]], repo)
        # Identical pip options and interpreter are used for both versions.
        execute(f"install_{phase}", [python, "-m", "pip", "install", "--force-reinstall", "."], repo)
        if phase == "reference":
            for name in ("extract", "analyze", "generate"):
                stage(name)
        stage(phase)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", required=True, help="run_id from the manifest")
    parser.add_argument("--root", type=Path, default=default_root(), help="parent of rq3_runs")
    parser.add_argument("--repository-root", type=Path, default=default_repository_root())
    parser.add_argument("--python", default=sys.executable, help="interpreter in the experiment environment")
    parser.add_argument("--k", type=int, default=500, help="V1 samples per API, as in main.py")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be positive")
    run_spec, manifest_dir = load_run_spec(args.manifest, args.run)
    run_dir = run_one(run_spec, manifest_dir, args.root, args.repository_root, args.python, args.k)
    print(f"Completed {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
