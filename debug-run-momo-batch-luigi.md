# Debug Session: run-momo-batch-luigi
- **Status**: [OPEN]
- **Issue**: Diagnose the current `run_momo_batch.py` defaults after switching the target library to Luigi.
- **Debug Server**: not started yet; first pass uses command output and existing artifacts. If code instrumentation is needed, use debug-server reporting before any business fix.
- **Log File**: pending

## Reproduction Steps
1. Launch `/Users/bytedance/Desktop/py_test/Momo_test/run_momo_batch.py` from the current workspace/default configuration.
2. Observe whether it fails during preflight, environment setup, dependency installation, `main.py --phase algo`, V1 probe/repair, or V2 replay.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| H1 | Current defaults point to Luigi but `LIB_FILE`, `LIB_NAME`, `LIB_GITNAME`, `START`, or `END` are inconsistent. | Medium | Low | Rejected: defaults are internally consistent: `luigi.txt`, `luigi`, `START=1`, `END=None`. |
| H2 | Luigi's required Python version is unavailable or not automatically discovered. | Medium | Low | Rejected: `uv` successfully provisions Python 3.8.3 for Luigi. |
| H3 | Luigi dependency normalization/install fails due to stale or unavailable pinned packages. | Medium | Medium | Confirmed: `pywin32==227`, `pydivert==2.1.0`, and `requests-async==0.5.0` are invalid on this macOS run; `ruamel.yaml.clib==0.2.0` source build also fails. |
| H4 | The LLM endpoint or parallel LLM calls fail during `main.py --phase algo`. | Medium | Low | Rejected so far: one-shot LLM probe succeeded; `stage_2_function.py` and `main.py` reached combination filtering. |
| H5 | V1 probe fails due to generated test harness or Luigi runtime side effects rather than setup. | Medium | Medium | Inconclusive: the controlled run was stopped during LLM combination filtering after setup began succeeding. |

## Log Evidence
- Current defaults: `LIB_FILE="luigi.txt"`, `LIB_GITNAME="luigi"`, `LIB_NAME="luigi"`, `START=1`, `END=None`, `DEFAULT_LLM_WORKERS=4`.
- Preflight passed for the current defaults: `Loaded 31 records`, selected `[1:None]`, first selected record is `bug_id=2`, Python `3.8.3`.
- LLM one-shot probe returned `MOMO_LUIGI_LINK_OK`.
- First one-record run failed during `uv pip install -r .../runs/luigi/2/environment/requirements.txt` because `pywin32==227` has only Windows wheels.
- Luigi requirements across the dataset include Windows-only packages: `pydivert==2.1.0`, `pypiwin32==223`, `pywin32==227`.
- After filtering Windows-only packages, the next resolver failure was `requests-async==0.5.0`, which is absent from PyPI and the configured mirror.
- After filtering `requests-async==0.5.0`, the next failure was `ruamel.yaml.clib==0.2.0` source build on macOS; `ruamel.yaml.clib==0.2.8` has a compatible cp38 macOS wheel.
- After rewriting `ruamel.yaml.clib==0.2.0` to `0.2.8`, dependency installation succeeded and the run reached `stage_2_function.py` and `main.py --phase algo`.
- Luigi bug 2 has 138 + 155 argument combinations before boundary generation; `check_condition_filter` was still a serial LLM stage, so it became the next practical bottleneck.
- `check_condition_filter` and `generate_api_boundary` were parallelized using the existing `MOMO_LLM_WORKERS` mechanism; unit tests pass with `MOMO_LLM_WORKERS=2`.

## Verification Conclusion
Current `run_momo_batch.py` is no longer blocked at preflight, LLM connectivity, or Luigi environment setup. The environment-normalization fixes allow Luigi bug 2 to reach algorithm execution. The remaining expensive area was stage-internal LLM calls in condition filtering and boundary generation; those stages now use parallel workers, matching the earlier test-case generation and repair model.
