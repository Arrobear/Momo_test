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


class RunApiFallbackTests(unittest.TestCase):
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
