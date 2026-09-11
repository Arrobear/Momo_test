import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rq3.diff_seed_generator import collect_implementation_diff
from rq3.json_values import json_key
from rq3.manifest import SOURCE_ROOT, load_run_spec
from rq3.run import DOCUMENTATION_DIRS, build_environment
from rq3.seed_loader import resolve_input_seed_bundle
from rq3.tests.helpers import make_repository


class CliTests(unittest.TestCase):
    def test_real_extract_and_analysis_stages_use_run_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run with spaces"
            root.mkdir()
            repo, reference, candidate = make_repository(root / "repositories")
            package_name = "rq3_cli_fixture"
            (repo / "src/demo").rename(repo / "src" / package_name)
            work = root / "work"
            work.mkdir()
            for name in DOCUMENTATION_DIRS:
                (root / "documentation" / name).mkdir(parents=True)
            run = {
                "run_id": "demo", "library": package_name, "repo_dir": "demo",
                "resolved_reference": reference, "resolved_candidate": candidate,
                "api_include": [package_name + ".echo"],
            }
            env = build_environment(run, root)
            # Import the synthetic src-layout fixture without installing test packages.
            env["PYTHONPATH"] += os.pathsep + str(repo / "src")
            env["MOMO_API_KEY"] = ""
            for phase in ("extract", "analyze"):
                result = subprocess.run(
                    [sys.executable, "-m", "rq3.stage", phase], cwd=work, env=env,
                    capture_output=True, text=True, encoding="utf-8", timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            sources = root / "documentation/api_src_code" / f"{package_name}_api_sources.json"
            data = json.loads(sources.read_text(encoding="utf-8"))
            self.assertIn("return str(value)", data[package_name + ".echo"]["python"]["code"])
            self.assertTrue((root / "documentation/arg_space" / f"{package_name}_arg_space_0.json").exists())

    def test_saved_response_cli_produces_loadable_commit_scoped_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, reference, candidate = make_repository(root / "repos")
            diff = collect_implementation_diff(repo, reference, candidate)
            hunk = next(line for line in diff["text"].splitlines() if line.startswith("@@"))
            run = {
                "run_id": "demo", "library": "demo", "repo_dir": "demo",
                "reference": reference, "candidate": candidate,
                "api_include": ["demo.echo"], "generated_input_seeds": "artifact.json",
                "diff_seed_generation": {"targets": {"demo.echo": ["value"]}},
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"runs": [run]}), encoding="utf-8")
            response = root / "response.json"
            values = [False, 0, 0.0, "  x  ", "list()", {"z": 1, "a": 0}]
            response.write_text(json.dumps({"candidates": [{
                "api": "demo.echo", "parameter": "value", "values": values,
                "rationale": "Exercise conversion inputs.",
                "diff_evidence": {"file": "src/demo/__init__.py", "hunk": hunk},
            }]}), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(SOURCE_ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
            result = subprocess.run([
                sys.executable, "-m", "rq3.diff_seed_generator", "--manifest", str(manifest),
                "--run", "demo", "--repository-root", str(root / "repos"),
                "--response-file", str(response), "--output", str(root / "artifact.json"),
            ], cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            spec, base = load_run_spec(manifest, "demo")
            spec.update(resolved_reference=reference, resolved_candidate=candidate)
            actual = resolve_input_seed_bundle(spec, base)["input_seeds"]["demo.echo"]["value"]
            self.assertEqual([json_key(v) for v in actual], [json_key(v) for v in values])

    def test_manifest_rejects_invalid_configuration_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            valid = {"run_id": "demo", "library": "demo", "repo_dir": "demo", "reference": "old", "candidate": "new"}
            for change in ({"repo_dir": "../other"}, {"candidate": None}, {"api_include": []}, {"api_include": ["x", "x"]}):
                with self.subTest(change=change):
                    path.write_text(json.dumps({"runs": [{**valid, **change}]}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_run_spec(path, "demo")


if __name__ == "__main__":
    unittest.main()
