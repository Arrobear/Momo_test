"""Execute upstream stages from <run>/work so ../documentation stays run-local."""

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

from rq3.api_filter import filter_api_definition
from rq3.json_values import load_json
from rq3.manifest import SOURCE_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "analyze", "generate", "reference", "candidate"))
    parser.add_argument("--k", type=int, default=500)
    args = parser.parse_args(argv)
    root = Path(os.environ["MOMO_ROOT_PATH"]).resolve()
    if Path.cwd().resolve().parent != root:
        raise ValueError("Stage cwd must be an immediate child of MOMO_ROOT_PATH")
    library = os.environ["MOMO_LIB_NAME"]
    api_file = root / "documentation" / "lib_api" / f"{library}_APIdef.txt"
    baseline = root / "documentation" / "results" / f"{library}_v1_baseline.json"
    scripts = {"extract": "extract_func_name.py", "analyze": "stage_2_function.py", "generate": "main.py"}
    if args.stage in scripts:
        script = SOURCE_ROOT / scripts[args.stage]
        sys.argv = [str(script), *(["--phase", "algo"] if args.stage == "generate" else [])]
        runpy.run_path(str(script), run_name="__main__")
        if args.stage == "extract":
            filter_api_definition(api_file)
            if not api_file.read_text(encoding="utf-8").strip():
                raise ValueError("No APIs were extracted")
        elif args.stage == "analyze":
            sources = root / "documentation" / "api_src_code" / f"{library}_api_sources.json"
            data = load_json(sources.read_text(encoding="utf-8"))
            names = [
                line.split("(", 1)[0].strip()
                for line in api_file.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            missing = [name for name in names if not data.get(name, {}).get("python", {}).get("code")]
            if missing:
                raise RuntimeError(f"Source resolver returned no Python source for: {missing}")
        return 0

    # Explicit dispatch avoids interpreting a leftover baseline as a phase selector.
    from stage_1_approch import run_test_cases_v1, run_test_cases_v2

    random.seed(42)
    if args.stage == "reference":
        if baseline.exists():
            raise FileExistsError(f"Reference baseline already exists: {baseline}")
        run_test_cases_v1(K=args.k, output_path=str(baseline))
        if not baseline.exists() or not load_json(baseline.read_text(encoding="utf-8")):
            raise RuntimeError("Reference stage produced no baseline cases")
    else:
        if not baseline.exists():
            raise FileNotFoundError(baseline)
        run_test_cases_v2(baseline_path=str(baseline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
