"""Exercise real upstream persistence and V1/V2 execution with synthetic APIs."""

import json
import os
import tempfile
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rq3.json_values import json_key
from rq3.seed_integration import decode_input_seed, merge_input_seeds


class StageRoundtripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import stage_1_approch
        import stage_1_function
        cls.stage = stage_1_approch
        cls.io = stage_1_function

    def test_resaving_candidates_keeps_each_seed_and_its_type_together(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            path = Path(directory) / "inputs.json"
            for seed in ["  x  ", "  y  ", "  x  "]:
                with patch.dict(os.environ, {"MOMO_INPUT_SEEDS": json.dumps({"demo.echo": {"value": [seed]}})}, clear=True):
                    candidates = merge_input_seeds("demo.echo", {"value": {
                        "type": "literal", "values": ["legacy"]
                    }}, {"value": "str"})
                self.io.save_api_inputs("demo.echo", candidates, str(path))
            saved = json.loads(path.read_text(encoding="utf-8"))["demo.echo"]["value"]
        self.assertEqual(saved["values"][0], "legacy")
        self.assertEqual([decode_input_seed(v)[1] for v in saved["values"][1:]], ["  x  ", "  y  "])
        self.assertNotIn("value_types", saved)
        for value in saved["values"][1:]:
            for kind in ("code", "literal"):
                decoded, ok = self.stage._eval_param_by_type(self.io.safe_serialize(value), kind)
                self.assertTrue(ok)
                self.assertEqual(decoded, decode_input_seed(value)[1])

    def test_no_seed_configuration_leaves_legacy_candidates_unchanged(self):
        candidates = {"value": {"type": "literal", "values": ["  x  ", "[1]", 0]}}
        before = json.dumps(candidates)
        with patch.dict(os.environ, {}, clear=True):
            merge_input_seeds("demo.echo", candidates, {"value": "object"})
        self.assertEqual(json.dumps(candidates), before)
        self.assertEqual(decode_input_seed("[1]"), (False, "[1]"))

    def test_stage_dispatch_rejects_leftover_reference_and_missing_candidate(self):
        from rq3.stage import main

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            original_cwd = Path.cwd()
            os.chdir(work)
            stack.callback(os.chdir, original_cwd)
            stack.enter_context(patch.dict(os.environ, {"MOMO_ROOT_PATH": str(root), "MOMO_LIB_NAME": "demo"}))
            reference = stack.enter_context(patch.object(self.stage, "run_test_cases_v1"))
            candidate = stack.enter_context(patch.object(self.stage, "run_test_cases_v2"))
            with self.assertRaises(FileNotFoundError):
                main(["candidate"])
            baseline = root / "documentation/results/demo_v1_baseline.json"
            baseline.parent.mkdir(parents=True)
            baseline.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                main(["reference"])
            reference.assert_not_called()
            candidate.assert_not_called()

    def test_real_v1_and_v2_preserve_json_inputs_and_dotted_import(self):
        values = ["  x  ", "[1]", "list()", False, 0, 0.0, True, 1, 1.0,
                  [False, 0, 0.0], {"text": "  y  ", "numbers": [True, 1]}]
        library = "rq3_fixture.api"
        api = library + ".echo"
        observed = {"v1": [], "v2": []}
        package = types.ModuleType("rq3_fixture")
        package.__path__ = []
        module = types.ModuleType(library)
        package.api = module

        def v1(value):
            observed["v1"].append(json_key(value))
            return "reference"

        def v2(value):
            observed["v2"].append(json_key(value))
            return "candidate"

        module.echo = v1
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            original_cwd = Path.cwd()
            os.chdir(work)
            stack.callback(os.chdir, original_cwd)
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(patch.dict("sys.modules", {"rq3_fixture": package, library: module}))
            for component in (self.stage, self.io):
                stack.enter_context(patch.object(component, "root_path", str(root)))
                stack.enter_context(patch.object(component, "lib_name", library))
            stack.enter_context(patch.object(self.stage, "_valid_params_cache", {}))
            stack.enter_context(patch.object(self.stage, "_eval_globals_cache", None))
            stack.enter_context(patch.dict(os.environ, {"MOMO_INPUT_SEEDS": json.dumps({api: {"value": values}})}, clear=True))
            docs = root / "documentation"
            (docs / "lib_api").mkdir(parents=True)
            (docs / "lib_api" / f"{library}_APIdef.txt").write_text(api + "(value)\n", encoding="utf-8")
            candidates = merge_input_seeds(api, {}, {"value": "object"})
            # Mixing a code-tagged parameter with JSON seeds used to execute strings.
            candidates["value"]["type"] = "code"
            self.io.save_api_inputs(api, candidates, str(docs / "api_input" / f"{library}_inputs_0.json"))
            self.io.save_api_inputs(api, f"def run_api(value):\n    return {api}(value)\n", str(docs / "test_cases" / f"{library}_case_0.json"))
            baseline = docs / "results" / f"{library}_v1_baseline.json"
            with patch.object(self.stage.random, "choices", side_effect=[[i] for i in range(len(values))]):
                self.stage.run_test_cases_v1(K=len(values), output_path=str(baseline))
            module.echo = v2
            report = docs / "results" / "diff.json"
            self.stage.run_test_cases_v2(baseline_path=str(baseline), report_path=str(report))
            records = json.loads(report.read_text(encoding="utf-8"))
        expected = [json_key(value) for value in values]
        self.assertEqual(observed["v1"], expected)
        self.assertEqual(observed["v2"], expected)
        self.assertEqual(len(records), len(values))


if __name__ == "__main__":
    unittest.main()
