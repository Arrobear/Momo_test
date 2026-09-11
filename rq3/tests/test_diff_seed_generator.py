import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from rq3.seed_loader import resolve_input_seed_bundle
from rq3.diff_seed_generator import (
    ARTIFACT_SCHEMA,
    build_artifact,
    build_prompt,
    collect_implementation_diff,
    is_implementation_path,
    normalize_llm_response,
)
from rq3.json_values import json_key


class DiffCollectionTests(unittest.TestCase):
    def git(self, repo, *arguments):
        subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_collects_source_diff_but_excludes_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            self.git(repo, "init")
            self.git(repo, "config", "user.email", "momo@example.invalid")
            self.git(repo, "config", "user.name", "Momo Test")
            (repo / "src").mkdir()
            (repo / "tests").mkdir()
            (repo / "src" / "demo.py").write_text(
                "def parse(value):\n    return value.strip()\n", encoding="utf-8"
            )
            (repo / "tests" / "test_demo.py").write_text(
                "def test_parse():\n    assert True\n", encoding="utf-8"
            )
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-m", "before")
            reference = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()

            (repo / "src" / "demo.py").write_text(
                "def parse(value):\n    return value.strip('\\x00')\n", encoding="utf-8"
            )
            (repo / "tests" / "test_demo.py").write_text(
                "def test_parse():\n    assert False\n", encoding="utf-8"
            )
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-m", "after")
            candidate = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()

            result = collect_implementation_diff(repo, reference, candidate)

        self.assertEqual(result["eligible_paths"], ["src/demo.py"])
        self.assertIn("tests/test_demo.py", result["excluded_paths"])
        self.assertIn("strip('\\x00')", result["text"])
        self.assertNotIn("assert False", result["text"])

    def test_excludes_single_file_test_modules(self):
        self.assertFalse(is_implementation_path("package/tests.py"))
        self.assertFalse(is_implementation_path("package/test_parser.py"))
        self.assertFalse(is_implementation_path("package/parser_tests.py"))
        self.assertFalse(is_implementation_path("package/_tests/parser.py"))
        self.assertTrue(is_implementation_path("package/parser.py"))


class LlmResponseValidationTests(unittest.TestCase):
    def response(self, **overrides):
        candidate = {
            "api": "demo.parse",
            "parameter": "value",
            "values": ["", "\u0000", "x\u0000"],
            "rationale": "Exercises the changed stripping boundary.",
            "diff_evidence": {"file": "src/demo.py", "hunk": "@@ -1 +1 @@"},
        }
        candidate.update(overrides)
        return json.dumps({"candidates": [candidate]})

    def test_normalizes_approved_candidates_to_input_seed_mapping(self):
        candidates, seeds = normalize_llm_response(
            self.response(),
            {"demo.parse": ["value"]},
            ["src/demo.py"],
            6,
            "+++ b/src/demo.py\n@@ -1 +1 @@\n",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(seeds["demo.parse"]["value"], ["", "\u0000", "x\u0000"])

    def test_response_preserves_boolean_integer_and_float_cases(self):
        values = [False, 0, 0.0, True, 1, 1.0]
        _, seeds = normalize_llm_response(
            self.response(values=values), {"demo.parse": ["value"]},
            ["src/demo.py"], 6, "+++ b/src/demo.py\n@@ -1 +1 @@\n",
        )
        self.assertEqual([json_key(v) for v in seeds["demo.parse"]["value"]], [json_key(v) for v in values])

    def test_prompt_lists_complete_allowed_hunk_headers(self):
        diff = (
            "+++ b/src/demo.py\n"
            "@@ -10,3 +10,4 @@ def parse(value):\n"
            "+    return value\n"
        )
        prompt = build_prompt(
            {"kind": "calibration"},
            {"demo.parse": ["value"]},
            diff,
            6,
        )
        self.assertIn(
            "- src/demo.py: @@ -10,3 +10,4 @@ def parse(value):",
            prompt,
        )

    def test_rejects_unapproved_parameter(self):
        with self.assertRaisesRegex(ValueError, "unapproved parameter"):
            normalize_llm_response(
                self.response(parameter="mode"),
                {"demo.parse": ["value"]},
                ["src/demo.py"],
                6,
                "+++ b/src/demo.py\n@@ -1 +1 @@\n",
            )

    def test_keeps_call_like_strings_as_json_data(self):
        _, seeds = normalize_llm_response(
            self.response(values=["demo.Factory()"]),
            {"demo.parse": ["value"]},
            ["src/demo.py"],
            6,
            "+++ b/src/demo.py\n@@ -1 +1 @@\n",
        )
        self.assertEqual(seeds["demo.parse"]["value"], ["demo.Factory()"])

    def test_rejects_hunk_not_present_in_diff(self):
        with self.assertRaisesRegex(ValueError, "hunk not present"):
            normalize_llm_response(
                self.response(),
                {"demo.parse": ["value"]},
                ["src/demo.py"],
                6,
                "+++ b/src/demo.py\n@@ -9 +9 @@\n",
            )

    def test_truncates_repeated_parameter_groups_to_aggregate_limit(self):
        first = json.loads(self.response())["candidates"][0]
        second = dict(first, values=["a", "b", "c"])
        response = json.dumps({"candidates": [first, second]})
        candidates, seeds = normalize_llm_response(
            response,
            {"demo.parse": ["value"]},
            ["src/demo.py"],
            5,
            "+++ b/src/demo.py\n@@ -1 +1 @@\n",
        )
        self.assertEqual(len(seeds["demo.parse"]["value"]), 5)
        self.assertEqual(candidates[-1]["values"], ["a", "b"])

    def test_artifact_records_raw_response_and_call_controls(self):
        response = self.response()
        artifact = build_artifact(
            {
                "run_id": "demo-run",
                "library": "demo",
                "reference": "old",
                "candidate": "new",
                "kind": "frontier",
            },
            {
                "reference_commit": "0" * 40,
                "candidate_commit": "1" * 40,
                "eligible_paths": ["src/demo.py"],
                "excluded_paths": ["tests/test_demo.py"],
                "text": "+++ b/src/demo.py\n@@ -1 +1 @@\n",
            },
            "prompt",
            response,
            "demo-model",
            "https://example.invalid",
            120.0,
            3,
            {"demo.parse": ["value"]},
            6,
        )
        self.assertEqual(artifact["generation"]["raw_response"], response)
        self.assertEqual(artifact["generation"]["timeout_seconds"], 120.0)
        self.assertEqual(artifact["input_seeds"]["demo.parse"]["value"][0], "")


class CampaignSeedLoaderTests(unittest.TestCase):
    def write_artifact(self, directory, **scope_overrides):
        scope = {
            "run_id": "demo-run",
            "reference": "old",
            "candidate": "new",
            "targets": {"demo.parse": ["value"]},
        }
        scope.update(scope_overrides)
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "scope": scope,
            "information_policy": {"mode": "implementation_diff_only"},
            "diff": {"sha256": "diff-hash"},
            "input_seeds": {"demo.parse": {"value": [0, 1]}},
        }
        path = Path(directory) / "seeds.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        return path

    def run_spec(self, artifact_path):
        return {
            "run_id": "demo-run",
            "reference": "old",
            "candidate": "new",
            "api_include": ["demo.parse"],
            "input_seeds": {"demo.parse": {"value": [-1, 0]}},
            "generated_input_seeds": str(artifact_path),
        }

    def test_merges_manual_and_generated_seeds_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_artifact(directory)
            bundle = resolve_input_seed_bundle(self.run_spec(path), directory)
        self.assertEqual(bundle["input_seeds"]["demo.parse"]["value"], [-1, 0, 1])
        self.assertEqual(bundle["sources"][1]["diff_sha256"], "diff-hash")

    def test_rejects_artifact_for_different_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_artifact(directory, candidate="other")
            with self.assertRaisesRegex(ValueError, "candidate mismatch"):
                resolve_input_seed_bundle(self.run_spec(path), directory)


if __name__ == "__main__":
    unittest.main()
