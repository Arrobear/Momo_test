import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rq3.api_filter import filter_api_definition, requested_api_names
from rq3.run import build_environment
from rq3.manifest import load_run_spec
from rq3.seed_integration import decode_input_seed, merge_input_seeds
from rq3.seed_loader import resolve_input_seed_bundle
from rq3.json_values import json_key


class ApiFilterTests(unittest.TestCase):
    def test_filters_in_requested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "APIdef.txt"
            path.write_text("demo.first(x)\ndemo.second(y)\n", encoding="utf-8")
            selected = filter_api_definition(path, ["demo.second", "demo.first"])
            self.assertEqual(selected, ["demo.second(y)", "demo.first(x)"])
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                "demo.second(y)\ndemo.first(x)\n",
            )

    def test_environment_filter_is_optional(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(requested_api_names())

    def test_keeps_duplicate_definitions_for_requested_api(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "APIdef.txt"
            path.write_text("demo.parse(x)\ndemo.parse(x, mode=None)\n", encoding="utf-8")
            selected = filter_api_definition(path, ["demo.parse"])
        self.assertEqual(selected, ["demo.parse(x)", "demo.parse(x, mode=None)"])


class SeedIntegrationTests(unittest.TestCase):
    def test_merges_seed_values_without_duplicates(self):
        raw = json.dumps({"demo.parse": {"value": [0, 1]}})
        candidates = {"value": {"type": "literal", "values": [-1, 0]}}
        with patch.dict(os.environ, {"MOMO_INPUT_SEEDS": raw}, clear=True):
            result = merge_input_seeds("demo.parse", candidates, {"value": "int"})
        self.assertEqual(result["value"]["values"][:2], [-1, 0])
        self.assertEqual([decode_input_seed(v)[1] for v in result["value"]["values"][2:]], [0, 1])
        self.assertNotIn("value_types", result["value"])

    def test_keeps_code_and_literal_interpretations_of_same_string(self):
        raw = json.dumps({"demo.parse": {"value": ["demo.Factory()"]}})
        candidates = {
            "value": {"type": "code", "values": ["demo.Factory()"]}
        }
        with patch.dict(os.environ, {"MOMO_INPUT_SEEDS": raw}, clear=True):
            result = merge_input_seeds("demo.parse", candidates, {"value": "object"})
        self.assertEqual(result["value"]["values"][0], "demo.Factory()")
        self.assertEqual(decode_input_seed(result["value"]["values"][1]), (True, "demo.Factory()"))

    def test_preserves_nested_json_types_and_object_key_order(self):
        values = [False, 0, 0.0, True, 1, 1.0, [False], [0], {"a": 0, "b": 1}]
        bundle = resolve_input_seed_bundle({"input_seeds": {
            "demo.parse": {"value": values + [{"b": 1, "a": 0}]}
        }}, ".")
        actual = bundle["input_seeds"]["demo.parse"]["value"]
        self.assertEqual([json_key(v) for v in actual], [json_key(v) for v in values + [{"b": 1, "a": 0}]])

    def test_rejects_nonfinite_manual_values(self):
        with self.assertRaises(ValueError):
            resolve_input_seed_bundle({"input_seeds": {"demo.parse": {"value": [float("nan")]}}}, ".")

    def test_rejects_unknown_parameter(self):
        raw = json.dumps({"demo.parse": {"missing": [0]}})
        with patch.dict(os.environ, {"MOMO_INPUT_SEEDS": raw}, clear=True):
            with self.assertRaisesRegex(ValueError, "unknown parameters"):
                merge_input_seeds("demo.parse", {}, {"value": "int"})


class RunConfigurationTests(unittest.TestCase):
    def test_loads_one_run_and_builds_environment(self):
        manifest = {
            "runs": [
                {
                    "run_id": "demo-run",
                    "library": "demo",
                    "import_name": "demo.api",
                    "repo_dir": "demo-repo",
                    "reference": "old",
                    "candidate": "new",
                    "api_include": ["demo.api.parse"],
                    "input_seeds": {"demo.api.parse": {"value": [0]}},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "runs.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            run_spec, base_dir = load_run_spec(manifest_path, "demo-run")
            run_spec.update(resolved_reference="a" * 40, resolved_candidate="b" * 40)
            with patch.dict(os.environ, {"MOMO_INPUT_SEEDS": "stale", "MOMO_API_INCLUDE": "stale"}, clear=True):
                environment = build_environment(run_spec, Path(directory) / "root")
        self.assertEqual(environment["MOMO_LIB_NAME"], "demo.api")
        self.assertEqual(environment["MOMO_REFERENCE_COMMIT"], "a" * 40)
        self.assertNotIn("MOMO_INPUT_SEEDS", environment)
        self.assertEqual(Path(environment["MOMO_INPUT_SEEDS_FILE"]).parent, Path(environment["MOMO_ROOT_PATH"]))


if __name__ == "__main__":
    unittest.main()
