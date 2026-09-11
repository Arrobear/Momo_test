import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rq3.manifest import SOURCE_ROOT
from rq3.run import run_command, run_one
from rq3.tests.helpers import git, make_repository


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo, reference, candidate = make_repository(self.root / "repos")
        self.spec = {
            "run_id": "demo-run", "library": "demo", "repo_dir": "demo",
            "reference": reference, "candidate": candidate,
            "api_include": ["demo.echo"], "input_seeds": {"demo.echo": {"value": [" x "]}},
        }

    def test_sequence_uses_isolated_clone_and_cwd_without_modifying_original(self):
        calls = []
        original_head = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "untracked.txt").write_text("preserve", encoding="utf-8")

        def execute(command, cwd, environment, log_path):
            calls.append((log_path.stem, command, cwd, environment))
            # Exercise actual Git cloning/checkouts; do not install into the test interpreter.
            if command[0] == "git":
                run_command(command, cwd, environment, log_path)
            elif log_path.stem == "install_reference":
                (cwd / "build").mkdir()
                (cwd / "build/stale.py").write_text("old build", encoding="utf-8")
            elif log_path.stem == "install_candidate":
                self.assertFalse((cwd / "build").exists())

        with patch("rq3.run.run_command", side_effect=execute):
            run_dir = run_one(self.spec, self.root, self.root / "output space", self.root / "repos", k=7)
        self.assertEqual([c[0] for c in calls], [
            "clone", "checkout_reference", "install_reference", "extract", "analyze", "generate",
            "reference", "clean_candidate", "checkout_candidate", "install_candidate", "candidate",
        ])
        for label, command, cwd, environment in calls:
            if label in {"extract", "analyze", "generate", "reference", "candidate"}:
                self.assertEqual(cwd.parent, Path(environment["MOMO_ROOT_PATH"]))
                self.assertEqual(command[-2:], ["--k", "7"])
        cloned_repo = run_dir / "repositories/demo"
        self.assertEqual(git(cloned_repo, "rev-parse", "HEAD"), original_head)
        self.assertFalse((cloned_repo / "untracked.txt").exists())
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), original_head)
        self.assertTrue((self.repo / "untracked.txt").exists())
        self.assertEqual(json.loads((run_dir / "input_seeds.json").read_text()), self.spec["input_seeds"])
        with self.assertRaises(FileExistsError):
            run_one(self.spec, self.root, self.root / "output space", self.root / "repos")

    def test_failed_stage_stops_before_candidate_and_preserves_attempt(self):
        labels = []

        def execute(command, cwd, environment, log_path):
            labels.append(log_path.stem)
            if log_path.stem == "generate":
                raise RuntimeError("generation failed")

        with patch("rq3.run.run_command", side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                run_one(self.spec, self.root, self.root / "out", self.root / "repos")
        self.assertNotIn("checkout_candidate", labels)
        self.assertTrue((self.root / "out/rq3_runs/demo-run/run.json").is_file())

    def test_rejects_artifact_for_a_moved_git_ref_before_starting(self):
        self.spec["candidate"] = "HEAD"
        path = self.root / "seeds.json"
        path.write_text(json.dumps({
            "schema": "momo.diff-seeds.v1",
            "scope": {**self.spec, "resolved_reference": self.spec["reference"], "resolved_candidate": "0" * 40},
            "input_seeds": {},
        }), encoding="utf-8")
        self.spec["generated_input_seeds"] = path.name
        with self.assertRaisesRegex(ValueError, "resolved_candidate mismatch"):
            run_one(self.spec, self.root, self.root / "out", self.root / "repos")
        self.assertFalse((self.root / "out").exists())

    def test_source_resolver_uses_configured_repository_directory(self):
        with patch.dict(os.environ, {"MOMO_REPOSITORY_ROOT": str(self.root / "repos")}):
            spec = importlib.util.spec_from_file_location("test_resolver", SOURCE_ROOT / "source_resolver.py")
            resolver = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(resolver)
        source = resolver.resolve_api_source("demo.echo", "demo")
        self.assertIn("return str(value)", source["source_code"])
        self.assertTrue(Path(source["file"]).is_relative_to(self.repo))


if __name__ == "__main__":
    unittest.main()
