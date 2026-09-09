import subprocess
import sys
import tempfile
import unittest
import json
import os
import platform
from pathlib import Path
from unittest import mock


MOMO_DIR = Path(__file__).resolve().parent
if str(MOMO_DIR) not in sys.path:
    sys.path.insert(0, str(MOMO_DIR))

import run_momo_batch as batch
import source_resolver
import stage_1_approch
import stage_2_function
import test_case_refiner
import test_environment
import test_executor


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

    def test_select_records_by_non_contiguous_bug_ids(self):
        records = [
            {"bug_id": 1},
            {"bug_id": 2},
            {"bug_id": 5},
            {"bug_id": 7},
        ]

        selected = batch.select_records(records, 0, 1, [7, 2])

        self.assertEqual([record["bug_id"] for record in selected], [7, 2])

    def test_parse_bug_ids_accepts_commas_and_spaces(self):
        self.assertEqual(batch.parse_bug_ids("2, 5 7"), [2, 5, 7])


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

    def test_baseline_summary_detects_framework_only_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "demo_v1_baseline.json"
            baseline.write_text(
                '{"demo.api": [{"函数运行状态": "error", '
                '"函数返回结果": "run_api 函数加载失败"}]}',
                encoding="utf-8",
            )
            summary = batch.summarize_baseline([baseline])

        self.assertEqual(summary["cases"], 1)
        self.assertEqual(summary["framework_load_failures"], 1)

    def test_static_path_summary_merges_pages_by_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arg_space = root / "documentation" / "arg_space"
            arg_space.mkdir(parents=True)
            (arg_space / "demo_arg_space_0.json").write_text(
                json.dumps(
                    {
                        "demo.first": [{"id": "P1"}, {"id": "P2"}],
                        "demo.replaced": [{"id": "old"}],
                    }
                ),
                encoding="utf-8",
            )
            (arg_space / "demo_arg_space_1.json").write_text(
                json.dumps(
                    {
                        "demo.replaced": [{"id": "new1"}, {"id": "new2"}],
                        "demo.last": [{"id": "P3"}],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(batch, "ROOT", root):
                summary = batch.summarize_static_paths("demo")

        self.assertEqual(summary["total"], 5)
        self.assertEqual(
            summary["apis"],
            {"demo.first": 2, "demo.replaced": 2, "demo.last": 1},
        )

    def test_record_is_skipped_after_stage_two_when_path_limit_is_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bug_repo = root / "bug-repo"
            bug_repo.mkdir()
            record = {
                "bug_id": 7,
                "python_version": "3.8.3",
                "bug_version": "bug",
                "fix_version": "fix",
                "bug_api": ["demo.call(value)"],
            }
            options = batch.build_parser().parse_args(
                [
                    "--lib-name",
                    "demo",
                    "--lib-gitname",
                    "demo",
                    "--joern",
                    "never",
                    "--max-paths-per-bug",
                    "4",
                ]
            )
            bug_environment = mock.MagicMock()
            fix_environment = mock.MagicMock()
            bug_environment.create.return_value = bug_environment
            fix_environment.create.return_value = fix_environment
            worktree = mock.MagicMock()
            worktree.__enter__.return_value = bug_repo
            worktree.__exit__.return_value = False

            metadata = {
                "bug_dir": root / "bug-data",
                "required_python": "3.8.3",
                "pythonpath": "",
                "requirements": root / "requirements.txt",
                "setup": root / "setup.sh",
            }
            with mock.patch.object(batch, "RUNTIME_DIR", root / "runtime"), \
                    mock.patch.object(
                        batch,
                        "bugsinpy_record_metadata",
                        return_value=metadata,
                    ), \
                    mock.patch.object(
                        batch,
                        "prepare_requirements",
                        return_value={},
                    ), \
                    mock.patch.object(batch, "cleanup_intermediate_files"), \
                    mock.patch.object(batch, "cleanup_transient_results"), \
                    mock.patch.object(batch, "write_api_defs"), \
                    mock.patch.object(
                        batch,
                        "resolve_test_python_for_record",
                        return_value=("local", Path(sys.executable)),
                    ), \
                    mock.patch.object(
                        batch,
                        "StrictTestEnvironment",
                        side_effect=[bug_environment, fix_environment],
                    ), \
                    mock.patch.object(
                        batch,
                        "managed_worktree",
                        return_value=worktree,
                    ), \
                    mock.patch.object(
                        batch,
                        "make_runtime_env",
                        return_value={},
                    ), \
                    mock.patch.object(
                        batch,
                        "summarize_static_paths",
                        return_value={"total": 5, "apis": {"demo.call": 5}},
                    ), \
                    mock.patch.object(batch, "run_python") as run_python:
                result = batch.process_record(record, root / "repo", options)

        self.assertFalse(result)
        self.assertEqual(run_python.call_count, 1)
        self.assertEqual(run_python.call_args.args[0], "stage_2_function.py")

    def test_algorithm_steps_use_runner_python_and_execution_uses_test_envs(self):
        class FakeTestEnvironment:
            def __init__(self, python, env_name):
                self.python = Path(python)
                self.env_name = env_name

            def create(self):
                return self

            def install_requirements(self, requirements_path, metadata):
                return None

            def install_requirements_without_dependencies(self, requirements):
                return None

            def run_setup(self, setup_path, worktree, extra_pythonpath=None):
                return None

            def install_target(self, repo_dir):
                return None

            def runtime_env(self, worktree, extra_pythonpath=None):
                return {"ENV_NAME": self.env_name}

            def manifest(self):
                return {
                    "python_executable": str(self.python),
                    "provider": "local",
                    "python_match": True,
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bug_repo = root / "bug-repo"
            fix_repo = root / "fix-repo"
            bundle = root / "bundle"
            for path in (bug_repo, fix_repo, bundle):
                path.mkdir(parents=True)

            runner_python = root / "runner" / "bin" / "python"
            bug_python = root / "bug-env" / "bin" / "python"
            fix_python = root / "fix-env" / "bin" / "python"
            for path in (runner_python, bug_python, fix_python):
                path.parent.mkdir(parents=True)
                path.write_text("", encoding="utf-8")

            record = {
                "bug_id": 8,
                "python_version": "3.8.3",
                "bug_version": "bug",
                "fix_version": "fix",
                "bug_api": ["demo.call(value)"],
            }
            options = batch.build_parser().parse_args(
                [
                    "--lib-name",
                    "demo",
                    "--lib-gitname",
                    "demo",
                    "--joern",
                    "never",
                ]
            )
            options.python = runner_python

            metadata = {
                "bug_dir": root / "bug-data",
                "required_python": "3.8.3",
                "pythonpath": "",
                "requirements": root / "requirements.txt",
                "setup": root / "setup.sh",
            }
            bug_environment = FakeTestEnvironment(bug_python, "bug")
            fix_environment = FakeTestEnvironment(fix_python, "fix")
            bug_context = mock.MagicMock()
            bug_context.__enter__.return_value = bug_repo
            bug_context.__exit__.return_value = False
            fix_context = mock.MagicMock()
            fix_context.__enter__.return_value = fix_repo
            fix_context.__exit__.return_value = False

            def run_python_side_effect(script, args=None, **kwargs):
                if script == "test_case_refiner.py":
                    status_path = Path(args[args.index("--status") + 1])
                    status_path.write_text(
                        json.dumps({"pending": 0, "total": 1}),
                        encoding="utf-8",
                    )

            with mock.patch.object(batch, "RUNTIME_DIR", root / "runtime"), \
                    mock.patch.object(
                        batch,
                        "bugsinpy_record_metadata",
                        return_value=metadata,
                    ), \
                    mock.patch.object(
                        batch,
                        "prepare_requirements",
                        return_value={},
                    ), \
                    mock.patch.object(batch, "cleanup_intermediate_files"), \
                    mock.patch.object(batch, "cleanup_transient_results"), \
                    mock.patch.object(batch, "write_api_defs"), \
                    mock.patch.object(
                        batch,
                        "resolve_test_python_for_record",
                        return_value=("local", bug_python),
                    ), \
                    mock.patch.object(
                        batch,
                        "StrictTestEnvironment",
                        side_effect=[bug_environment, fix_environment],
                    ), \
                    mock.patch.object(
                        batch,
                        "managed_worktree",
                        side_effect=[bug_context, fix_context],
                    ), \
                    mock.patch.object(
                        batch,
                        "make_runtime_env",
                        return_value={"RUNNER_ENV": "1"},
                    ), \
                    mock.patch.object(
                        batch,
                        "summarize_static_paths",
                        return_value={"total": 1, "apis": {"demo.call": 1}},
                    ), \
                    mock.patch.object(
                        batch,
                        "prepare_test_bundle",
                        return_value=bundle,
                    ), \
                    mock.patch.object(
                        batch,
                        "prune_unvalidated_path_cases",
                        return_value={"validated": 1, "pruned": 0},
                    ), \
                    mock.patch.object(batch, "publish_executor_results"), \
                    mock.patch.object(batch, "move_results"), \
                    mock.patch.object(
                        batch,
                        "run_python",
                        side_effect=run_python_side_effect,
                    ) as run_python, \
                    mock.patch.object(batch, "run") as run_command:
                self.assertTrue(batch.process_record(record, root / "repo", options))

        algorithm_scripts = [
            call.args[0] for call in run_python.call_args_list
        ]
        self.assertEqual(
            algorithm_scripts,
            ["stage_2_function.py", "main.py", "test_case_refiner.py"],
        )
        self.assertTrue(
            all(
                call.kwargs["python_executable"] == runner_python
                for call in run_python.call_args_list
            )
        )

        executor_commands = [
            call.args[0]
            for call in run_command.call_args_list
            if str(call.args[0][1]).endswith("test_executor.py")
        ]
        self.assertEqual(len(executor_commands), 3)
        self.assertEqual(executor_commands[0][0], str(bug_python))
        self.assertIn("--mode", executor_commands[0])
        self.assertEqual(executor_commands[0][executor_commands[0].index("--mode") + 1], "probe")
        self.assertEqual(executor_commands[1][0], str(bug_python))
        self.assertEqual(executor_commands[1][executor_commands[1].index("--mode") + 1], "v1")
        self.assertEqual(executor_commands[2][0], str(fix_python))
        self.assertEqual(executor_commands[2][executor_commands[2].index("--mode") + 1], "v2")

    def test_path_case_bundle_does_not_require_legacy_flat_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_case_dir = root / "documentation" / "test_cases"
            input_dir = root / "documentation" / "api_input"
            test_case_dir.mkdir(parents=True)
            input_dir.mkdir(parents=True)
            path_case = {
                "schema_version": 2,
                "case_id": "demo.add_P1::case_1",
                "code": "def run_test_case():\n    pass\n",
            }
            (test_case_dir / "demo_case_0.json").write_text(
                json.dumps({"demo.add": [path_case]}),
                encoding="utf-8",
            )
            record = {
                "bug_id": 1,
                "python_version": "3.8.3",
                "bug_version": "bug",
                "fix_version": "fix",
                "bug_api": ["demo.add(left, right)"],
            }

            with mock.patch.object(batch, "ROOT", root):
                bundle = batch.prepare_test_bundle(
                    root / "run",
                    "demo",
                    record,
                )

            bundled_cases = json.loads(
                (bundle / "test_cases.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                bundled_cases["demo.add"][0]["case_id"],
                "demo.add_P1::case_1",
            )
            self.assertEqual(
                json.loads(
                    (bundle / "inputs.json").read_text(encoding="utf-8")
                ),
                {},
            )

    def test_prune_unvalidated_path_cases_keeps_validated_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            cases = {
                "demo.add": [
                    {
                        "schema_version": 2,
                        "case_id": "demo.add_P1::case_1",
                        "validated": True,
                    },
                    {
                        "schema_version": 2,
                        "case_id": "demo.add_P2::case_1",
                        "validated": False,
                    },
                ],
                "demo.sub": [
                    {
                        "schema_version": 2,
                        "case_id": "demo.sub_P1::case_1",
                        "validated": True,
                    }
                ],
            }
            (bundle / "test_cases.json").write_text(
                json.dumps(cases), encoding="utf-8"
            )

            summary = batch.prune_unvalidated_path_cases(bundle)

            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["validated"], 2)
            self.assertEqual(summary["pruned"], 1)
            pruned_cases = json.loads(
                (bundle / "test_cases.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [case["case_id"] for case in pruned_cases["demo.add"]],
                ["demo.add_P1::case_1"],
            )
            self.assertEqual(
                summary["apis"]["demo.add"]["pruned_case_ids"],
                ["demo.add_P2::case_1"],
            )
            self.assertTrue((bundle / "pruned_cases.json").exists())


class WorktreeIsolationTests(unittest.TestCase):
    def test_project_repo_url_reads_bugsinpy_project_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "matplotlib"
            project_dir.mkdir(parents=True)
            (project_dir / "project.info").write_text(
                'github_url="https://github.com/matplotlib/matplotlib"\n',
                encoding="utf-8",
            )

            with mock.patch.object(batch, "BUGSINPY_DIR", root):
                self.assertEqual(
                    batch._project_repo_url("matplotlib"),
                    "https://github.com/matplotlib/matplotlib",
                )

    def test_ensure_repo_backs_up_source_snapshot_before_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dl_lib = root / "dl_lib"
            bugsinpy = root / "bugsinpy"
            repo_dir = dl_lib / "matplotlib"
            repo_dir.mkdir(parents=True)
            (repo_dir / "setup.py").write_text("# snapshot\n", encoding="utf-8")
            project_dir = bugsinpy / "matplotlib"
            project_dir.mkdir(parents=True)
            (project_dir / "project.info").write_text(
                'github_url="https://github.com/matplotlib/matplotlib"\n',
                encoding="utf-8",
            )

            def fake_run(command, **kwargs):
                self.assertEqual(command[0:2], ["git", "clone"])
                cloned = Path(command[-1])
                cloned.mkdir(parents=True)
                (cloned / ".git").mkdir()

            with mock.patch.object(batch, "DL_LIB_DIR", dl_lib), \
                    mock.patch.object(batch, "BUGSINPY_DIR", bugsinpy), \
                    mock.patch.object(batch, "run", side_effect=fake_run):
                resolved = batch.ensure_repo("matplotlib")

            self.assertEqual(resolved, repo_dir)
            self.assertTrue((dl_lib / "matplotlib" / ".git").exists())
            self.assertTrue(
                (dl_lib / "matplotlib.source-snapshot" / "setup.py").exists()
            )

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

    def test_restore_git_symlinks_replaces_materialized_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            link_parent = repo / "lib" / "ansible" / "module_utils"
            link_parent.mkdir(parents=True)
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
            (repo / "lib" / "ansible" / "release.py").write_text(
                "__version__ = 'test'\n",
                encoding="utf-8",
            )
            link_path = link_parent / "ansible_release.py"
            os.symlink("../release.py", str(link_path))
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "symlink"], cwd=repo, check=True)

            link_path.unlink()
            link_path.write_text("../release.py", encoding="utf-8")

            restored = batch.restore_git_symlinks(repo)

            self.assertEqual(
                restored,
                ["lib/ansible/module_utils/ansible_release.py"],
            )
            self.assertTrue(link_path.is_symlink())
            self.assertEqual(os.readlink(str(link_path)), "../release.py")


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


class RunApiFallbackTests(unittest.TestCase):
    def test_resolve_api_callable_strips_source_layout_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "demo_pkg"
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "api.py").write_text(
                "def call():\n"
                "    return 'ok'\n",
                encoding="utf-8",
            )
            sys.path.insert(0, tmp)
            try:
                resolved = test_executor.resolve_api_callable(
                    "lib.demo_pkg.api.call"
                )
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved(), "ok")
                self.assertEqual(resolved.__module__, "demo_pkg.api")
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("demo_pkg.api", None)
                sys.modules.pop("demo_pkg", None)

    def test_load_run_api_falls_back_to_real_callable(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_path = Path(tmp) / "demo_target.py"
            module_path.write_text(
                "def add(left, right):\n"
                "    return left + right\n",
                encoding="utf-8",
            )
            sys.path.insert(0, tmp)
            try:
                with mock.patch.object(
                    stage_1_approch, "read_json_api", return_value=None
                ):
                    run_api = stage_1_approch._load_run_api("demo_target.add")
                self.assertIsNotNone(run_api)
                self.assertEqual(run_api(left=2, right=3), 5)
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("demo_target", None)


class StrictEnvironmentTests(unittest.TestCase):
    def test_python_version_must_include_exact_patch(self):
        with self.assertRaises(test_environment.TestEnvironmentError):
            test_environment.normalize_python_version("3.8")

    def test_wrong_python_version_is_rejected(self):
        with self.assertRaises(test_environment.TestEnvironmentError):
            test_environment.assert_exact_python(sys.executable, "0.0.1")

    def test_query_python_version_ignores_pth_noise_after_json(self):
        noisy_output = (
            '{"version": "3.8.1", "executable": "/tmp/python"}\n'
            "Error processing line 1 of matplotlib-nspkg.pth:\n"
            "Remainder of file ignored\n"
        )
        with mock.patch.object(
            test_environment.subprocess,
            "check_output",
            return_value=noisy_output,
        ):
            info = test_environment.query_python_version("/tmp/python")

        self.assertEqual(info["version"], "3.8.1")
        self.assertEqual(info["executable"], "/tmp/python")

    def test_llm_worker_count_is_configurable(self):
        with mock.patch.dict(os.environ, {"MOMO_LLM_WORKERS": "3"}):
            self.assertEqual(stage_1_approch.get_llm_worker_count(), 3)
        with mock.patch.dict(os.environ, {"MOMO_LLM_WORKERS": "invalid"}):
            self.assertEqual(stage_1_approch.get_llm_worker_count(), 4)
        with mock.patch.dict(os.environ, {"MOMO_LLM_WORKERS": "0"}):
            self.assertEqual(stage_1_approch.get_llm_worker_count(), 1)

    def test_condition_combination_batch_maps_results_to_original_order(self):
        combinations = [["required"], ["required", "optional"]]
        task = {
            "batch_index": 0,
            "start_index": 8,
            "fun_string": "demo.call",
            "arg_combinations": combinations,
            "api_def": "demo.call(required, optional=None)",
            "api_doc": "Demo documentation.",
        }
        response = json.dumps(
            {
                "results": [
                    {"index": 0, "valid": True},
                    {"index": 1, "valid": False},
                ]
            }
        )

        with mock.patch.object(
            stage_1_approch,
            "_thread_llm_client",
            return_value=object(),
        ), mock.patch.object(
            stage_1_approch,
            "call_llm_with_retry",
            return_value=response,
        ) as call_llm:
            result = stage_1_approch._check_condition_combination_batch(task)

        self.assertEqual(
            result,
            [
                (8, combinations[0], False),
                (9, combinations[1], True),
            ],
        )
        call_llm.assert_called_once()
        prompt = call_llm.call_args.kwargs["messages"][1]["content"]
        self.assertEqual(prompt.count("Demo documentation."), 1)
        self.assertIn('"index": 0', prompt)
        self.assertIn('"index": 1', prompt)

    def test_condition_filter_sends_combinations_in_batches(self):
        combinations = [
            ["required"],
            ["required", "first"],
            ["required", "second"],
        ]
        responses = [
            json.dumps(
                {
                    "results": [
                        {"index": 0, "valid": True},
                        {"index": 1, "valid": False},
                    ]
                }
            ),
            json.dumps({"results": [{"index": 0, "valid": True}]}),
        ]

        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data="demo.call(required, first=None)\n"),
        ), mock.patch.object(
            stage_1_approch,
            "get_all_combinations_from_json",
            return_value=(combinations, 0),
        ), mock.patch.object(
            stage_1_approch,
            "get_doc",
            return_value="Demo documentation.",
        ), mock.patch.object(
            stage_1_approch,
            "get_llm_worker_count",
            return_value=1,
        ), mock.patch.object(
            stage_1_approch,
            "get_llm_combination_batch_size",
            return_value=2,
        ), mock.patch.object(
            stage_1_approch,
            "_thread_llm_client",
            return_value=object(),
        ), mock.patch.object(
            stage_1_approch,
            "call_llm_with_retry",
            side_effect=responses,
        ) as call_llm, mock.patch.object(
            stage_1_approch,
            "append_filtered_combinations_to_json",
        ) as append_results:
            stage_1_approch.check_condition_filter(["demo.call"])

        self.assertEqual(call_llm.call_count, 2)
        self.assertEqual(
            append_results.call_args.args[2],
            [["required", "first"]],
        )

    def test_large_parameter_space_is_bounded_before_llm_filter(self):
        args = ["path", "endpoint"] + ["optional_%d" % index for index in range(20)]
        conditions = {
            "Mandatory Parameters": ["path", "endpoint"],
            "Mutually Exclusive Parameter Pairs": [],
            "Mandatory Coexistence Parameters": [],
        }

        combinations = stage_1_approch.generate_bounded_parameter_combinations(
            args,
            conditions,
            max_combinations=256,
        )

        self.assertLessEqual(len(combinations), 256)
        self.assertLess(len(combinations), 1048576)
        self.assertIn(["path", "endpoint"], combinations)
        self.assertIn(args, combinations)

    def test_boundary_combinations_are_capped_per_path(self):
        combinations = [
            ["required"] + [f"optional_{index}"]
            for index in range(20)
        ]
        combinations.append(["required"])
        combinations.append(["required"] + [f"optional_{index}" for index in range(20)])

        selected = stage_1_approch.select_representative_combinations(
            combinations,
            max_combinations=4,
        )

        self.assertLessEqual(len(selected), 4)
        self.assertIn(["required"], selected)
        self.assertIn(
            ["required"] + [f"optional_{index}" for index in range(20)],
            selected,
        )

    def test_static_path_space_is_capped_and_deduplicated(self):
        paths = []
        for index in range(100):
            paths.append(
                {
                    "id": f"old_{index}",
                    "path_type": "return" if index % 2 else "raise",
                    "conjuncts": [f"x == {index}", "worker is not None"],
                    "complexity": index % 9,
                }
            )
        paths.append(
            {
                "id": "duplicate",
                "path_type": "raise",
                "conjuncts": ["x == 0", "worker is not None"],
                "complexity": 0,
            }
        )

        selected = stage_2_function.select_representative_paths(
            paths,
            "demo.api",
            max_paths=8,
        )

        signatures = {
            (path["path_type"], tuple(path["conjuncts"]))
            for path in selected
        }
        self.assertLessEqual(len(selected), 8)
        self.assertEqual(len(signatures), len(selected))
        self.assertIn("raise", {path["path_type"] for path in selected})
        self.assertIn("return", {path["path_type"] for path in selected})
        self.assertEqual(selected[0]["id"], "demo.api_1")

    def test_parser_does_not_hardcode_test_python(self):
        options = batch.build_parser().parse_args([])
        self.assertEqual(options.test_env_provider, "conda")
        self.assertIsNone(options.test_python)
        self.assertEqual(options.llm_workers, 4)
        self.assertEqual(
            options.max_paths_per_bug,
            batch.DEFAULT_MAX_PATHS_PER_BUG,
        )
        self.assertFalse(options.create_conda_envs)

    def test_conda_provider_uses_existing_exact_python(self):
        version = ".".join(map(str, sys.version_info[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            env_dir = Path(tmp) / "envs" / ("momo-py%s" % version)
            python_path = env_dir / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("", encoding="utf-8")
            options = batch.build_parser().parse_args([])

            with mock.patch.object(
                batch,
                "_conda_command",
                return_value="/tmp/conda",
            ), mock.patch.object(
                batch,
                "_conda_env_prefixes",
                return_value=[env_dir],
            ), mock.patch.object(
                batch,
                "assert_exact_python",
                return_value={"version": version, "executable": str(python_path)},
            ), mock.patch.object(batch, "run") as run_command:
                provider, python = batch.resolve_test_python_for_record(
                    version,
                    options,
                )

        self.assertEqual(provider, "local")
        self.assertEqual(python, python_path.resolve())
        run_command.assert_not_called()

    def test_conda_provider_creates_missing_seed_environment(self):
        version = "3.8.3"
        with tempfile.TemporaryDirectory() as tmp:
            options = batch.build_parser().parse_args(["--create-conda-envs"])
            options.conda_env_root = Path(tmp) / "conda-envs"
            options.conda_subdir = "osx-64"
            expected_python = (
                options.conda_env_root / "python-3.8.3" / "bin" / "python"
            )

            with mock.patch.object(
                batch,
                "_conda_command",
                return_value="/tmp/conda",
            ), mock.patch.object(
                batch,
                "_conda_env_prefixes",
                return_value=[],
            ), mock.patch.object(
                batch,
                "assert_exact_python",
                return_value={"version": version, "executable": str(expected_python)},
            ), mock.patch.object(batch, "run") as run_command:
                provider, python = batch.resolve_test_python_for_record(
                    version,
                    options,
                )

        self.assertEqual(provider, "local")
        self.assertEqual(python, expected_python.resolve())
        command = run_command.call_args.args[0]
        self.assertEqual(
            command[:5],
            [
                "/tmp/conda",
                "create",
                "-y",
                "-p",
                str(options.conda_env_root / "python-3.8.3"),
            ],
        )
        self.assertIn("python=3.8.3", command)
        self.assertEqual(run_command.call_args.kwargs["env"]["CONDA_SUBDIR"], "osx-64")

    def test_project_seed_python_is_discovered_by_required_version(self):
        version = ".".join(map(str, sys.version_info[:3]))
        version_digits = "".join(version.split("."))
        with tempfile.TemporaryDirectory() as tmp:
            seed_python = Path(tmp) / (
                "py%s-test" % version_digits
            ) / "bin" / "python"
            seed_python.parent.mkdir(parents=True)
            os.symlink(sys.executable, str(seed_python))

            with mock.patch.dict(
                os.environ,
                {"MOMO_PYTHON_SEED_DIR": tmp},
            ):
                candidates = list(
                    test_environment._local_python_candidates(version)
                )

        self.assertIn(Path(sys.executable).resolve(), candidates)

    def test_prepare_requirements_filters_target_package_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "requirements.txt"
            destination = Path(tmp) / "normalized.txt"
            source.write_text(
                "\n".join(
                    [
                        "ansible-base==2.10.0.dev0",
                        "argcomplete==1.11.1",
                        "Jinja2==3.0.0a1",
                        "pkg-resources==0.0.0",
                        "pydivert==2.1.0",
                        "pypiwin32==223",
                        "pywin32==227",
                        "PythonLabs==1.0.2",
                        "requests-async==0.5.0",
                        "mysql-connector-python==8.0.19",
                        "mysql-connector-python==8.0.20",
                        "numpy==1.19.0rc2",
                        "scipy==1.5.0rc1",
                        "ruamel.yaml.clib==0.2.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            metadata = test_environment.prepare_requirements(
                source,
                destination,
                {"ansible", "ansible-base", "ansible-core"},
            )

            normalized = destination.read_text(encoding="utf-8")
            self.assertNotIn("ansible-base", normalized)
            self.assertNotIn("pkg-resources", normalized)
            self.assertNotIn("pydivert", normalized)
            self.assertNotIn("pypiwin32", normalized)
            self.assertNotIn("pywin32", normalized)
            self.assertNotIn("requests-async", normalized)
            self.assertIn("argcomplete==1.11.1", normalized)
            self.assertIn("Jinja2==3.0.0a1", normalized)
            self.assertIn("mysql-connector-python==8.0.21", normalized)
            self.assertIn("numpy==1.19.0", normalized)
            self.assertIn("scipy==1.5.0", normalized)
            self.assertIn("ruamel.yaml.clib==0.2.8", normalized)
            self.assertNotIn("mysql-connector-python==8.0.19", normalized)
            self.assertNotIn("mysql-connector-python==8.0.20", normalized)
            self.assertNotIn("numpy==1.19.0rc2", normalized)
            self.assertNotIn("scipy==1.5.0rc1", normalized)
            self.assertNotIn("ruamel.yaml.clib==0.2.0", normalized)
            self.assertEqual(
                metadata["removed_local_project_lines"],
                ["ansible-base==2.10.0.dev0"],
            )
            self.assertEqual(
                metadata["removed_unsupported_lines"],
                [
                    "pkg-resources==0.0.0",
                    "pydivert==2.1.0",
                    "pypiwin32==223",
                    "pywin32==227",
                    "PythonLabs==1.0.2",
                    "requests-async==0.5.0",
                ],
            )
            self.assertEqual(
                metadata["rewritten_unavailable_lines"],
                [
                    {
                        "from": "mysql-connector-python==8.0.19",
                        "to": "mysql-connector-python==8.0.21",
                    },
                    {
                        "from": "mysql-connector-python==8.0.20",
                        "to": "mysql-connector-python==8.0.21",
                    },
                    {"from": "numpy==1.19.0rc2", "to": "numpy==1.19.0"},
                    {"from": "scipy==1.5.0rc1", "to": "scipy==1.5.0"},
                    {
                        "from": "ruamel.yaml.clib==0.2.0",
                        "to": "ruamel.yaml.clib==0.2.8",
                    },
                ],
            )

    def test_prepare_requirements_adds_project_extra_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "requirements.txt"
            destination = Path(tmp) / "normalized.txt"
            source.write_text("numpy==1.19.5\n", encoding="utf-8")

            metadata = test_environment.prepare_requirements(
                source,
                destination,
                {"matplotlib"},
                extra_requirements=[
                    "numpy==1.19.5",
                    "cycler>=0.10",
                    "pyparsing>=2.0.1,<3",
                ],
            )

            normalized = destination.read_text(encoding="utf-8")
            self.assertEqual(normalized.count("numpy==1.19.5"), 1)
            self.assertIn("cycler>=0.10", normalized)
            self.assertIn("pyparsing>=2.0.1,<3", normalized)
            self.assertEqual(
                metadata["added_project_requirement_lines"],
                ["cycler>=0.10", "pyparsing>=2.0.1,<3"],
            )

    def test_matplotlib_extra_requirements_are_declared(self):
        self.assertEqual(
            batch.project_extra_requirements("matplotlib", "matplotlib"),
            [
                "numpy==1.19.5",
                "cycler>=0.10",
                "kiwisolver>=1.0.1",
                "pyparsing>=2.0.1,<3",
                "python-dateutil>=2.1",
                "pillow>=6.2.0",
            ],
        )

    def test_sanic_extra_requirements_restore_archived_requests_async(self):
        requirements = batch.project_extra_requirements("sanic", "sanic")

        self.assertIn("ujson==4.3.0", requirements)
        self.assertEqual(
            batch.project_no_deps_requirements("sanic", "sanic"),
            [
                "git+https://github.com/encode/requests-async.git"
                "@614f40f77f19e6c6da8a212ae799107b0384dbf9"
            ],
        )
        self.assertFalse(batch.should_run_project_setup("sanic", "sanic"))
        self.assertTrue(batch.should_run_project_setup("pandas", "pandas"))

    def test_no_deps_requirements_use_isolated_python(self):
        version = ".".join(map(str, sys.version_info[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            environment = test_environment.StrictTestEnvironment(
                version,
                Path(tmp) / "env",
                sys.executable,
            )
            environment.python = Path(sys.executable)

            with mock.patch.object(test_environment, "_run_checked") as run_checked, \
                    mock.patch.object(test_environment, "assert_exact_python"):
                environment.install_requirements_without_dependencies(
                    ["git+https://example.invalid/archive.git@abc123"]
                )

            command = run_checked.call_args.args[0]
            self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
            self.assertIn("--no-deps", command)

    def test_matplotlib_worktree_gets_system_freetype_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / "worktree"
            worktree.mkdir()

            result = batch.prepare_target_worktree_for_install(
                "matplotlib",
                "matplotlib",
                worktree,
            )

            self.assertEqual(result, worktree / "setup.cfg")
            config = result.read_text(encoding="utf-8")
            self.assertIn("system_freetype = True", config)
            self.assertIn("tests = False", config)

    def test_pandas_worktree_disables_werror_for_modern_clang(self):
        with tempfile.TemporaryDirectory() as tmp:
            worktree = Path(tmp) / "worktree"
            worktree.mkdir()
            setup_py = worktree / "setup.py"
            setup_py.write_text(
                'extra_compile_args = ["-Werror"]\n',
                encoding="utf-8",
            )

            result = batch.prepare_target_worktree_for_install(
                "pandas",
                "pandas",
                worktree,
            )

            self.assertEqual(result, setup_py)
            text = setup_py.read_text(encoding="utf-8")
            self.assertIn("extra_compile_args = []", text)
            self.assertIn("disabled -Werror", text)
            self.assertNotIn('extra_compile_args = ["-Werror"]', text)

    def test_install_env_exposes_conda_base_build_paths(self):
        version = ".".join(map(str, sys.version_info[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "env"
            worktree = root / "worktree"
            base_prefix = root / "conda" / "envs" / "py"
            (base_prefix / "bin").mkdir(parents=True)
            (base_prefix / "lib" / "pkgconfig").mkdir(parents=True)
            (base_prefix / "include").mkdir(parents=True)
            worktree.mkdir()
            environment = test_environment.StrictTestEnvironment(
                version,
                env_dir,
                sys.executable,
            )
            environment.python = Path(sys.executable)

            with mock.patch.object(
                test_environment,
                "_python_base_prefix",
                return_value=base_prefix,
            ):
                env = environment.install_env(worktree)

            self.assertTrue(
                env["PATH"].startswith(str(base_prefix / "bin") + os.pathsep)
            )
            self.assertEqual(
                env["PKG_CONFIG_PATH"],
                str(base_prefix / "lib" / "pkgconfig"),
            )
            self.assertEqual(env["CPATH"], str(base_prefix / "include"))
            self.assertEqual(env["LIBRARY_PATH"], str(base_prefix / "lib"))

    def test_upgrade_installer_tools_uses_compatible_caps(self):
        with mock.patch.object(test_environment, "_run_checked") as run_checked:
            test_environment._upgrade_installer_tools(
                Path("/tmp/python"),
                "https://pypi.tuna.tsinghua.edu.cn/simple",
            )

        command = run_checked.call_args.args[0]
        self.assertEqual(command[:4], ["/tmp/python", "-m", "pip", "install"])
        self.assertIn("--upgrade", command)
        self.assertIn("pip<25", command)
        self.assertIn("setuptools<70", command)
        self.assertIn("wheel<0.43", command)
        self.assertIn("pypi.tuna.tsinghua.edu.cn", command)

    def test_runtime_env_isolates_home_cache_and_ansible_temp(self):
        version = ".".join(map(str, sys.version_info[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "env"
            worktree = root / "worktree"
            worktree.mkdir()
            environment = test_environment.StrictTestEnvironment(
                version,
                env_dir,
                sys.executable,
            )

            env = environment.runtime_env(worktree)

            self.assertEqual(env["HOME"], str(env_dir.resolve() / "home"))
            self.assertEqual(
                env["ANSIBLE_LOCAL_TEMP"],
                str(env_dir.resolve() / "home" / ".ansible" / "tmp"),
            )
            self.assertEqual(env["XDG_CACHE_HOME"], str(env_dir.resolve() / "cache"))
            self.assertTrue(Path(env["ANSIBLE_LOCAL_TEMP"]).is_dir())

    def test_install_target_uses_isolated_runtime_env(self):
        version = ".".join(map(str, sys.version_info[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "env"
            worktree = root / "worktree"
            worktree.mkdir()
            (worktree / "setup.py").write_text("", encoding="utf-8")
            environment = test_environment.StrictTestEnvironment(
                version,
                env_dir,
                sys.executable,
            )
            environment.python = Path(sys.executable)

            with mock.patch.object(test_environment, "_run_checked") as run_checked, \
                    mock.patch.object(test_environment, "assert_exact_python"):
                environment.install_target(worktree)

            kwargs = run_checked.call_args.kwargs
            self.assertEqual(kwargs["cwd"], str(worktree.resolve()))
            self.assertEqual(kwargs["env"]["HOME"], str(env_dir.resolve() / "home"))
            self.assertEqual(
                kwargs["env"]["ANSIBLE_LOCAL_TEMP"],
                str(env_dir.resolve() / "home" / ".ansible" / "tmp"),
            )
            self.assertIn("--no-deps", run_checked.call_args.args[0])

    def test_pyproject_target_install_uses_non_editable_build(self):
        version = ".".join(map(str, sys.version_info[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "env"
            worktree = root / "worktree"
            worktree.mkdir()
            (worktree / "pyproject.toml").write_text(
                "[build-system]\nrequires = ['flit']\n",
                encoding="utf-8",
            )
            environment = test_environment.StrictTestEnvironment(
                version,
                env_dir,
                sys.executable,
                provider="uv",
            )
            environment.python = Path(sys.executable)
            environment.provider_used = "uv"
            environment.uv_command = ["uv"]

            with mock.patch.dict(os.environ, {"PYTHONPATH": "/outer/uv/path"}), \
                    mock.patch.object(test_environment, "_run_checked") as run_checked, \
                    mock.patch.object(test_environment, "assert_exact_python"):
                environment.install_target(worktree)

            command = run_checked.call_args.args[0]
            env = run_checked.call_args.kwargs["env"]
            self.assertNotIn("--editable", command)
            self.assertIn("--no-deps", command)
            self.assertEqual(command[-1], str(worktree.resolve()))
            self.assertNotIn("__PYVENV_LAUNCHER__", env)
            self.assertEqual(env["PYTHONPATH"], "/outer/uv/path")

    def test_setup_script_target_install_is_skipped(self):
        version = ".".join(map(str, sys.version_info[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = root / "setup.sh"
            worktree = root / "worktree"
            worktree.mkdir()
            setup.write_text("python setup.py develop\n", encoding="utf-8")
            environment = test_environment.StrictTestEnvironment(
                version,
                root / "env",
                sys.executable,
            )

            with mock.patch.object(test_environment, "_run_checked") as run_checked:
                environment.run_setup(setup, worktree)

            run_checked.assert_not_called()

    def test_setup_script_crlf_is_normalized_before_execution(self):
        version = ".".join(map(str, sys.version_info[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_dir = root / "env"
            worktree = root / "worktree"
            env_dir.mkdir()
            worktree.mkdir()
            setup = root / "setup.sh"
            setup.write_bytes(b"echo preparing\r\npip install helper-package\r\n")
            environment = test_environment.StrictTestEnvironment(
                version,
                env_dir,
                sys.executable,
            )
            environment.python = Path(sys.executable)

            executed = {}

            def capture(command, **kwargs):
                executed["command"] = command
                executed["content"] = Path(command[1]).read_bytes()

            with mock.patch.object(
                test_environment, "_run_checked", side_effect=capture
            ), mock.patch.object(test_environment, "assert_exact_python"):
                environment.run_setup(setup, worktree)

            self.assertEqual(
                executed["content"],
                b"echo preparing\npip install helper-package\n",
            )
            self.assertEqual(executed["command"][0], "/bin/bash")
            self.assertFalse((env_dir / ".momo_setup.sh").exists())

    def test_minimal_executor_replays_same_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            bundle = root / "bundle"
            results = root / "results"
            target.mkdir()
            bundle.mkdir()
            (target / "demo_target.py").write_text(
                "def add(left, right):\n"
                "    return left + right\n",
                encoding="utf-8",
            )
            (bundle / "record.json").write_text(
                json.dumps(
                    {
                        "lib_name": "demo_target",
                        "api_names": ["demo_target.add"],
                    }
                ),
                encoding="utf-8",
            )
            (bundle / "inputs.json").write_text(
                json.dumps(
                    {
                        "demo_target.add": {
                            "left": {"type": "literal", "values": [1, 2]},
                            "right": {"type": "literal", "values": [3, 4]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (bundle / "test_cases.json").write_text("{}", encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(target)
            version = platform.python_version()
            executor = MOMO_DIR / "test_executor.py"
            for mode in ("v1", "v2"):
                subprocess.run(
                    [
                        sys.executable,
                        str(executor),
                        "--mode",
                        mode,
                        "--bundle",
                        str(bundle),
                        "--results",
                        str(results),
                        "--expected-python",
                        version,
                        "--k",
                        "3",
                    ],
                    env=env,
                    check=True,
                )

            baseline = json.loads(
                (results / "v1_baseline.json").read_text(encoding="utf-8")
            )
            differences = json.loads(
                (results / "diff_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(baseline["demo_target.add"]), 3)
            self.assertEqual(differences, [])


class PathCaseGenerationTests(unittest.TestCase):
    def test_generates_k_complete_cases_for_every_static_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api_dir = root / "documentation" / "lib_api"
            api_dir.mkdir(parents=True)
            (api_dir / "demo_APIdef.txt").write_text(
                "demo.add(left, right)\n",
                encoding="utf-8",
            )
            bug_dir = root / "bug"
            bug_dir.mkdir()
            (bug_dir / "bug_patch.txt").write_text(
                "diff --git a/demo.py b/demo.py\n+handle failure\n",
                encoding="utf-8",
            )
            (bug_dir / "run_test.sh").write_text(
                "python -m unittest demo_test\n",
                encoding="utf-8",
            )
            paths = [
                {
                    "id": "demo.add_P1",
                    "conjuncts": ["left >= 0"],
                    "path_type": "return",
                    "src": ["python"],
                },
                {
                    "id": "demo.add_P2",
                    "conjuncts": ["left < 0"],
                    "path_type": "raise",
                    "src": ["python"],
                },
            ]

            def read_artifact(api_name, file_path, read_mode):
                return {
                    "src_code": {"python": {"code": "def add(): pass"}},
                    "conditions": {"Parameter type": {"left": "int"}},
                    "boundary": [],
                    "arg_space": paths,
                }[read_mode]

            generated = json.dumps(
                {
                    "code": (
                        "def run_test_case():\n"
                        "    import demo\n"
                        "    return momo_call(demo.add, 1, 2)\n"
                    ),
                    "summary": "fixed integers",
                }
            )
            with mock.patch.object(stage_1_approch, "root_path", str(root)), \
                    mock.patch.object(stage_1_approch, "lib_name", "demo"), \
                    mock.patch.object(
                        stage_1_approch, "make_client", return_value=object()
                    ), \
                    mock.patch.object(
                        stage_1_approch, "get_doc", return_value="demo doc"
                    ), \
                    mock.patch.object(
                        stage_1_approch,
                        "read_json_api",
                        side_effect=read_artifact,
                    ), \
                    mock.patch.object(
                        stage_1_approch,
                        "call_llm_with_retry",
                        return_value=generated,
                    ), \
                    mock.patch.dict(
                        os.environ,
                        {
                            "MOMO_REQUIRED_PYTHON": "3.8.3",
                            "MOMO_BUG_ID": "7",
                            "MOMO_BUG_DIR": str(bug_dir),
                        },
                    ):
                stage_1_approch.generate_test_cases(["demo.add"], k=2)

            cases = json.loads(
                (
                    root
                    / "documentation"
                    / "test_cases"
                    / "demo_case_0.json"
                ).read_text(encoding="utf-8")
            )["demo.add"]
            self.assertEqual(len(cases), 4)
            self.assertEqual(
                [case["expected_status"] for case in cases],
                ["success", "success", "error", "error"],
            )
            self.assertEqual(
                {case["required_python"] for case in cases},
                {"3.8.3"},
            )
            self.assertEqual(
                len({case["case_id"] for case in cases}),
                4,
            )
            self.assertIn("handle failure", cases[0]["bug_context"]["bug_patch"])
            self.assertIn("demo_test", cases[0]["bug_context"]["run_test"])


class PathCaseFeedbackTests(unittest.TestCase):
    def test_path_case_can_use_target_callable_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "demo_callable.py").write_text(
                "def double(value):\n"
                "    return value * 2\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(target))
            try:
                observation = test_executor.execute_path_case(
                    "demo_callable.double",
                    "demo_callable",
                    {
                        "case_id": "demo_callable.double_P1::case_1",
                        "summary": "target placeholder",
                        "code": (
                            "def run_test_case():\n"
                            "    return momo_call(target_callable, 4)\n"
                        ),
                    },
                    timeout=2,
                )
                self.assertEqual(observation["函数运行状态"], "success")
                self.assertEqual(observation["函数返回结果"], 8)
                self.assertTrue(observation["target_matches_api"])
            finally:
                sys.path.remove(str(target))
                sys.modules.pop("demo_callable", None)

    def test_path_case_restores_process_platform_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "demo_state.py").write_text(
                "def identity(value):\n"
                "    return value\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(target))
            original_platform = sys.platform
            try:
                observation = test_executor.execute_path_case(
                    "demo_state.identity",
                    "demo_state",
                    {
                        "case_id": "demo_state.identity_P1::case_1",
                        "summary": "platform mutation",
                        "code": (
                            "def run_test_case():\n"
                            "    import sys\n"
                            "    import demo_state\n"
                            "    sys.platform = 'win32'\n"
                            "    return momo_call(demo_state.identity, 7)\n"
                        ),
                    },
                    timeout=2,
                )
                self.assertEqual(observation["函数运行状态"], "success")
                self.assertGreater(
                    observation["target_trace"]["executed_line_count"],
                    0,
                )
                self.assertEqual(sys.platform, original_platform)
            finally:
                sys.path.remove(str(target))
                sys.modules.pop("demo_state", None)

    def test_success_with_none_result_is_not_a_valid_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "demo_none.py").write_text(
                "def touch(value):\n"
                "    value.append('done')\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(target))
            try:
                case_data = {
                    "case_id": "demo_none.touch_P1::case_1",
                    "summary": "returns none",
                    "expected_status": "success",
                    "code": (
                        "def run_test_case():\n"
                        "    import demo_none\n"
                        "    values = []\n"
                        "    return momo_call(demo_none.touch, values)\n"
                    ),
                }
                observation = test_executor.execute_path_case(
                    "demo_none.touch",
                    "demo_none",
                    case_data,
                    timeout=2,
                )
                self.assertEqual(observation["函数运行状态"], "success")
                self.assertIsNone(observation["函数返回结果"])
                issues = test_case_refiner.automatic_issues(
                    case_data, observation
                )
                self.assertTrue(
                    any("structured oracle" in issue for issue in issues)
                )
            finally:
                sys.path.remove(str(target))
                sys.modules.pop("demo_none", None)

    def test_bug_patch_target_error_can_be_validated_by_model(self):
        case_data = {
            "expected_status": "success",
            "bug_context": {
                "bug_patch": "diff --git a/demo.py b/demo.py\n+except OSError"
            },
        }
        observation = {
            "函数运行状态": "error",
            "execution_phase": "target_error",
            "target_invoked": True,
            "target_call_count": 1,
            "target_matches_api": True,
            "target_trace": {"executed_line_count": 3},
            "target_completed": False,
            "函数返回结果": "OSError: simulated failure",
        }
        self.assertEqual(
            test_case_refiner.automatic_issues(case_data, observation),
            [],
        )
        self.assertTrue(
            test_executor.observation_matches_validated_case(
                case_data, observation
            )
        )

        no_patch = dict(case_data)
        no_patch["bug_context"] = {}
        self.assertTrue(
            any(
                "expected success" in issue
                for issue in test_case_refiner.automatic_issues(
                    no_patch, observation
                )
            )
        )

    def test_target_timeout_is_validated_without_llm_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            bundle = root / "bundle"
            results = root / "results"
            target.mkdir()
            bundle.mkdir()
            (target / "demo_timeout.py").write_text(
                "def spin():\n"
                "    while True:\n"
                "        pass\n",
                encoding="utf-8",
            )
            record = {
                "lib_name": "demo_timeout",
                "api_names": ["demo_timeout.spin"],
            }
            case_code = (
                "def run_test_case():\n"
                "    import demo_timeout\n"
                "    return momo_call(demo_timeout.spin)\n"
            )
            cases = {
                "demo_timeout.spin": [
                    {
                        "schema_version": 2,
                        "case_id": "demo_timeout.spin_P1::case_1",
                        "api_name": "demo_timeout.spin",
                        "api_signature": "demo_timeout.spin()",
                        "path_id": "demo_timeout.spin_P1",
                        "path_type": "return",
                        "path_constraints": [],
                        "expected_status": "success",
                        "code": case_code,
                        "summary": "target timeout",
                        "revision": 0,
                        "validated": False,
                        "validation_history": [],
                    }
                ]
            }
            (bundle / "record.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            (bundle / "inputs.json").write_text("{}", encoding="utf-8")
            (bundle / "test_cases.json").write_text(
                json.dumps(cases), encoding="utf-8"
            )

            sys.path.insert(0, str(target))
            try:
                test_executor.run_probe(bundle, results, timeout=0.2)
                attempts = json.loads(
                    (results / "v1_attempts.json").read_text(
                        encoding="utf-8"
                    )
                )
                observation = attempts["demo_timeout.spin"][0]
                self.assertEqual(observation["函数运行状态"], "timeout")
                self.assertEqual(
                    observation["execution_phase"], "target_timeout"
                )
                self.assertEqual(
                    test_case_refiner.automatic_issues(
                        cases["demo_timeout.spin"][0], observation
                    ),
                    [],
                )

                with mock.patch.object(
                    test_case_refiner, "call_llm_with_retry"
                ) as call_llm:
                    status = test_case_refiner.refine_cases(
                        bundle,
                        results / "v1_attempts.json",
                        round_number=1,
                        client=object(),
                    )
                call_llm.assert_not_called()
                self.assertEqual(status["pending"], 0)
                self.assertEqual(status["validated"], 1)

                test_executor.run_v1(bundle, results, count=1, timeout=0.2)
                baseline = json.loads(
                    (results / "v1_baseline.json").read_text(encoding="utf-8")
                )
                baseline_case = baseline["demo_timeout.spin"][0]
                self.assertEqual(baseline_case["函数运行状态"], "timeout")
                self.assertEqual(
                    baseline_case["execution_phase"], "target_timeout"
                )
                timeout_bugs = json.loads(
                    (results / "timeout_bugs.json").read_text(encoding="utf-8")
                )
                self.assertIn("demo_timeout.spin", timeout_bugs)
                self.assertEqual(
                    timeout_bugs["demo_timeout.spin"][0]["case_index"],
                    0,
                )
            finally:
                sys.path.remove(str(target))
                sys.modules.pop("demo_timeout", None)

    def test_invalid_setup_is_repaired_validated_and_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            bundle = root / "bundle"
            results = root / "results"
            target.mkdir()
            bundle.mkdir()
            (target / "demo_target.py").write_text(
                "def add(left, right):\n"
                "    return left + right\n",
                encoding="utf-8",
            )
            record = {
                "lib_name": "demo_target",
                "api_names": ["demo_target.add"],
            }
            initial_code = (
                "def run_test_case():\n"
                "    return open('missing-input.txt').read()\n"
            )
            repaired_code = (
                "def run_test_case():\n"
                "    import demo_target\n"
                "    return momo_call(demo_target.add, 2, 3)\n"
            )
            cases = {
                "demo_target.add": [
                    {
                        "schema_version": 2,
                        "case_id": "demo_target.add_P1::case_1",
                        "api_name": "demo_target.add",
                        "api_signature": "demo_target.add(left, right)",
                        "api_documentation": "",
                        "api_source": {},
                        "parameter_conditions": {},
                        "boundary_context": [],
                        "path_id": "demo_target.add_P1",
                        "path_type": "return",
                        "path_constraints": [],
                        "expected_status": "success",
                        "code": initial_code,
                        "summary": "",
                        "revision": 0,
                        "validated": False,
                        "validation_history": [],
                    }
                ]
            }
            (bundle / "record.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            (bundle / "inputs.json").write_text("{}", encoding="utf-8")
            (bundle / "test_cases.json").write_text(
                json.dumps(cases), encoding="utf-8"
            )

            sys.path.insert(0, str(target))
            try:
                test_executor.run_probe(bundle, results, timeout=2)
                attempts = json.loads(
                    (results / "v1_attempts.json").read_text(encoding="utf-8")
                )
                first = attempts["demo_target.add"][0]
                self.assertEqual(first["函数运行状态"], "harness_error")
                self.assertFalse(first["target_invoked"])

                repair_response = json.dumps(
                    {
                        "valid": False,
                        "reason": "setup failed before target invocation",
                        "repaired_code": repaired_code,
                        "summary": "add two integers",
                    }
                )
                with mock.patch.object(
                    test_case_refiner,
                    "call_llm_with_retry",
                    return_value=repair_response,
                ):
                    status = test_case_refiner.refine_cases(
                        bundle,
                        results / "v1_attempts.json",
                        round_number=1,
                        client=object(),
                    )
                self.assertEqual(status["pending"], 1)
                self.assertEqual(status["repaired"], 1)

                test_executor.run_probe(bundle, results, timeout=2)
                attempts = json.loads(
                    (results / "v1_attempts.json").read_text(encoding="utf-8")
                )
                second = attempts["demo_target.add"][0]
                self.assertEqual(second["函数运行状态"], "success")
                self.assertTrue(second["target_invoked"])
                self.assertTrue(second["target_matches_api"])
                self.assertEqual(second["target_call_count"], 1)
                self.assertEqual(second["函数返回结果"], 5)

                valid_response = json.dumps(
                    {
                        "valid": True,
                        "reason": "target return path executed successfully",
                        "repaired_code": None,
                        "summary": "add two integers",
                    }
                )
                with mock.patch.object(
                    test_case_refiner,
                    "call_llm_with_retry",
                    return_value=valid_response,
                ):
                    status = test_case_refiner.refine_cases(
                        bundle,
                        results / "v1_attempts.json",
                        round_number=2,
                        client=object(),
                    )
                self.assertEqual(status["pending"], 0)

                test_executor.run_v1(bundle, results, count=1, timeout=2)
                baseline = json.loads(
                    (results / "v1_baseline.json").read_text(encoding="utf-8")
                )
                baseline_case = baseline["demo_target.add"][0]
                self.assertEqual(
                    baseline_case["test_case_code"],
                    repaired_code.strip(),
                )
                self.assertEqual(baseline_case["函数返回结果"], 5)
                self.assertEqual(baseline_case["revision"], 1)

                test_executor.run_v2(bundle, results, timeout=2)
                differences = json.loads(
                    (results / "diff_report.json").read_text(encoding="utf-8")
                )
                self.assertEqual(differences, [])

                target_module = sys.modules["demo_target"]
                original_add = target_module.add
                target_module.add = lambda left, right: left + right + 1
                try:
                    test_executor.run_v2(bundle, results, timeout=2)
                    differences = json.loads(
                        (results / "diff_report.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(len(differences), 1)
                    self.assertEqual(
                        differences[0]["case_id"],
                        "demo_target.add_P1::case_1",
                    )
                    self.assertEqual(
                        differences[0]["v2_target_call_count"], 1
                    )
                    self.assertTrue(
                        differences[0]["v2_target_matches_api"]
                    )
                finally:
                    target_module.add = original_add
            finally:
                sys.path.remove(str(target))
                sys.modules.pop("demo_target", None)

    def test_refiner_reviews_multiple_pending_cases_with_configured_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            results = root / "results"
            bundle.mkdir()
            results.mkdir()
            record = {
                "lib_name": "demo_target",
                "api_names": ["demo_target.add"],
            }
            cases = {
                "demo_target.add": [
                    {
                        "schema_version": 2,
                        "case_id": "demo_target.add_P1::case_1",
                        "api_name": "demo_target.add",
                        "api_signature": "demo_target.add(left, right)",
                        "api_documentation": "",
                        "api_source": {},
                        "parameter_conditions": {},
                        "boundary_context": [],
                        "path_id": "demo_target.add_P1",
                        "path_type": "return",
                        "path_constraints": [],
                        "expected_status": "success",
                        "code": (
                            "def run_test_case():\n"
                            "    return momo_call(target_callable, 1, 2)\n"
                        ),
                        "summary": "",
                        "revision": 0,
                        "validated": False,
                        "validation_history": [],
                    },
                    {
                        "schema_version": 2,
                        "case_id": "demo_target.add_P2::case_1",
                        "api_name": "demo_target.add",
                        "api_signature": "demo_target.add(left, right)",
                        "api_documentation": "",
                        "api_source": {},
                        "parameter_conditions": {},
                        "boundary_context": [],
                        "path_id": "demo_target.add_P2",
                        "path_type": "return",
                        "path_constraints": [],
                        "expected_status": "success",
                        "code": (
                            "def run_test_case():\n"
                            "    return momo_call(target_callable, 3, 4)\n"
                        ),
                        "summary": "",
                        "revision": 0,
                        "validated": False,
                        "validation_history": [],
                    },
                ]
            }
            attempts = {
                "demo_target.add": [
                    {
                        "case_id": "demo_target.add_P1::case_1",
                        "函数运行状态": "success",
                        "函数返回结果": 3,
                        "execution_phase": "target",
                        "target_invoked": True,
                        "target_call_count": 1,
                        "target_completed": True,
                        "target_matches_api": True,
                        "target_trace": {"executed_line_count": 1},
                    },
                    {
                        "case_id": "demo_target.add_P2::case_1",
                        "函数运行状态": "success",
                        "函数返回结果": 7,
                        "execution_phase": "target",
                        "target_invoked": True,
                        "target_call_count": 1,
                        "target_completed": True,
                        "target_matches_api": True,
                        "target_trace": {"executed_line_count": 1},
                    },
                ]
            }
            (bundle / "record.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            (bundle / "inputs.json").write_text("{}", encoding="utf-8")
            (bundle / "test_cases.json").write_text(
                json.dumps(cases), encoding="utf-8"
            )
            (results / "v1_attempts.json").write_text(
                json.dumps(attempts), encoding="utf-8"
            )
            valid_response = json.dumps(
                {
                    "valid": True,
                    "reason": "path trace and oracle are valid",
                    "repaired_code": None,
                    "summary": "validated",
                }
            )

            with mock.patch.dict(os.environ, {"MOMO_LLM_WORKERS": "2"}), \
                    mock.patch.object(
                        test_case_refiner, "make_client", return_value=object()
                    ), \
                    mock.patch.object(
                        test_case_refiner,
                        "call_llm_with_retry",
                        return_value=valid_response,
                    ) as call_llm:
                status = test_case_refiner.refine_cases(
                    bundle,
                    results / "v1_attempts.json",
                    round_number=1,
                )

            self.assertEqual(call_llm.call_count, 2)
            self.assertEqual(status["validated"], 2)
            self.assertEqual(status["pending"], 0)
            updated = json.loads(
                (bundle / "test_cases.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [
                    case["case_id"]
                    for case in updated["demo_target.add"]
                    if case.get("validated")
                ],
                [
                    "demo_target.add_P1::case_1",
                    "demo_target.add_P2::case_1",
                ],
            )

    def test_async_target_must_be_driven_after_direct_momo_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "demo_async.py").write_text(
                "async def add(left, right):\n"
                "    return left + right\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(target))
            try:
                unawaited = {
                    "case_id": "demo_async.add_P1::case_1",
                    "summary": "unawaited async call",
                    "expected_status": "success",
                    "code": (
                        "def run_test_case():\n"
                        "    import demo_async\n"
                        "    return momo_call(demo_async.add, 2, 3)\n"
                    ),
                }
                observation = test_executor.execute_path_case(
                    "demo_async.add",
                    "demo_async",
                    unawaited,
                    timeout=2,
                )
                self.assertEqual(
                    observation["函数运行状态"], "harness_error"
                )
                self.assertEqual(
                    observation["execution_phase"], "target_not_awaited"
                )

                awaited = dict(unawaited)
                awaited["code"] = (
                    "def run_test_case():\n"
                    "    import asyncio\n"
                    "    import demo_async\n"
                    "    loop = asyncio.new_event_loop()\n"
                    "    try:\n"
                    "        awaitable = momo_call(demo_async.add, 2, 3)\n"
                    "        return loop.run_until_complete(awaitable)\n"
                    "    finally:\n"
                    "        loop.close()\n"
                )
                observation = test_executor.execute_path_case(
                    "demo_async.add",
                    "demo_async",
                    awaited,
                    timeout=2,
                )
                self.assertEqual(observation["函数运行状态"], "success")
                self.assertEqual(observation["函数返回结果"], 5)
                self.assertTrue(observation["target_matches_api"])
                self.assertEqual(
                    test_case_refiner.automatic_issues(
                        awaited, observation
                    ),
                    [],
                )
            finally:
                sys.path.remove(str(target))
                sys.modules.pop("demo_async", None)

    def test_raise_path_requires_exception_from_requested_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            (target / "demo_raise.py").write_text(
                "def reject(value):\n"
                "    if value < 0:\n"
                "        raise ValueError('negative value')\n"
                "    return value\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(target))
            try:
                case_data = {
                    "case_id": "demo_raise.reject_P1::case_1",
                    "summary": "negative value",
                    "expected_status": "error",
                    "code": (
                        "def run_test_case():\n"
                        "    import demo_raise\n"
                        "    return momo_call(demo_raise.reject, -1)\n"
                    ),
                }
                observation = test_executor.execute_path_case(
                    "demo_raise.reject",
                    "demo_raise",
                    case_data,
                    timeout=2,
                )
                self.assertEqual(observation["函数运行状态"], "error")
                self.assertEqual(observation["execution_phase"], "target_error")
                self.assertTrue(observation["target_matches_api"])
                self.assertEqual(
                    test_case_refiner.automatic_issues(
                        case_data, observation
                    ),
                    [],
                )

                wrong_target = dict(case_data)
                wrong_target["code"] = (
                    "def run_test_case():\n"
                    "    return momo_call(len, [])\n"
                )
                observation = test_executor.execute_path_case(
                    "demo_raise.reject",
                    "demo_raise",
                    wrong_target,
                    timeout=2,
                )
                issues = test_case_refiner.automatic_issues(
                    wrong_target, observation
                )
                self.assertTrue(
                    any("other than the requested API" in item for item in issues)
                )
            finally:
                sys.path.remove(str(target))
                sys.modules.pop("demo_raise", None)


if __name__ == "__main__":
    unittest.main()
