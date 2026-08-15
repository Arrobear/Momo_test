import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MOMO_DIR = Path(__file__).resolve().parent
if str(MOMO_DIR) not in sys.path:
    sys.path.insert(0, str(MOMO_DIR))

import run_momo_batch as batch
import source_resolver
import stage_1_approch


class ParseDatasetTests(unittest.TestCase):
    def test_parse_lib_file_keeps_complete_api_signatures(self):
        content = """\
bug_id: 7
python_version: 3.9
buggy: abc123
fixed: def456
bug_api:
    src.demo.api.call(value, optional=None):
--------
bug_id: invalid
buggy: abc
fixed: def
bug_api:
    demo.invalid():
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.txt"
            path.write_text(content, encoding="utf-8")
            records = batch.parse_lib_file(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["bug_id"], 7)
        self.assertEqual(
            records[0]["bug_api"],
            ["src.demo.api.call(value, optional=None):"],
        )


class ArtifactLifecycleTests(unittest.TestCase):
    def test_cleanup_removes_all_pages_but_not_other_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "documentation" / "api_input"
            target_dir.mkdir(parents=True)
            for name in (
                "demo_inputs_0.json",
                "demo_inputs_12.json",
                "demo_default_inputs_3.json",
                "other_inputs_0.json",
            ):
                (target_dir / name).write_text("{}", encoding="utf-8")

            with mock.patch.object(batch, "ROOT", root):
                batch.cleanup_intermediate_files("demo")

            self.assertFalse((target_dir / "demo_inputs_0.json").exists())
            self.assertFalse((target_dir / "demo_inputs_12.json").exists())
            self.assertFalse((target_dir / "demo_default_inputs_3.json").exists())
            self.assertTrue((target_dir / "other_inputs_0.json").exists())


class WorktreeIsolationTests(unittest.TestCase):
    def test_managed_worktree_does_not_touch_dirty_primary_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            runtime = root / "runtime"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(
                ["git", "config", "user.email", "momo@example.invalid"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Momo Test"],
                cwd=repo,
                check=True,
            )
            tracked = repo / "value.txt"
            tracked.write_text("v1\n", encoding="utf-8")
            subprocess.run(["git", "add", "value.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "v1"], cwd=repo, check=True)
            v1 = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            tracked.write_text("v2\n", encoding="utf-8")
            subprocess.run(["git", "commit", "-qam", "v2"], cwd=repo, check=True)
            tracked.write_text("local change\n", encoding="utf-8")

            with mock.patch.object(batch, "RUNTIME_DIR", runtime):
                with batch.managed_worktree(repo, v1, "demo") as worktree:
                    self.assertEqual(
                        (worktree / "value.txt").read_text(encoding="utf-8"),
                        "v1\n",
                    )
                    worktree_path = worktree

            self.assertEqual(tracked.read_text(encoding="utf-8"), "local change\n")
            self.assertFalse(worktree_path.exists())


class SourceResolverTests(unittest.TestCase):
    def test_resolves_src_layout_class_method_by_full_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            module = repo / "src" / "demo_lib" / "api.py"
            module.parent.mkdir(parents=True)
            module.write_text(
                "class Worker:\n"
                "    def execute(self, value):\n"
                "        return value\n",
                encoding="utf-8",
            )

            with mock.patch.object(source_resolver, "TARGET_REPO", str(repo)):
                source_resolver._cache.clear()
                result = source_resolver.resolve_api_source(
                    "demo_lib.api.Worker.execute", "demo"
                )
                source_resolver._cache.clear()

            self.assertIsNotNone(result)
            self.assertEqual(result["api_type"], "function")
            self.assertIn("def execute", result["source_code"])


class DifferentialModeTests(unittest.TestCase):
    def test_explicit_modes_do_not_depend_on_baseline_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(stage_1_approch, "root_path", tmp), \
                    mock.patch.object(stage_1_approch, "lib_name", "demo"), \
                    mock.patch.object(stage_1_approch, "run_test_cases_v1") as run_v1, \
                    mock.patch.object(stage_1_approch, "run_test_cases_v2") as run_v2, \
                    mock.patch.object(stage_1_approch, "generate_bug_report"):
                stage_1_approch.run_test_cases(K=3, mode="v1")
                run_v1.assert_called_once()
                run_v2.assert_not_called()

                run_v1.reset_mock()
                stage_1_approch.run_test_cases(K=3, mode="v2")
                run_v2.assert_called_once()
                run_v1.assert_not_called()


if __name__ == "__main__":
    unittest.main()
