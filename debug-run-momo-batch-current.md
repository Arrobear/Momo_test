# Debug Session: run-momo-batch-current
- **Status**: [OPEN]
- **Issue**: Diagnose the current behavior of `run_momo_batch.py` when launched from the IDE/current defaults.
- **Debug Server**: not started yet; first pass uses command output and existing artifacts. If code instrumentation is needed, use debug-server reporting before any business fix.
- **Log File**: pending

## Reproduction Steps
1. Launch `/Users/bytedance/Desktop/py_test/Momo_test/run_momo_batch.py` from the current workspace/default configuration.
2. Observe whether it fails before analysis, during environment setup, during `main.py --phase algo`, during LLM calls, during V1 probe/repair, or during V2 replay.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| H1 | Current defaults in `run_momo_batch.py` point to a library/index range that is inconsistent with the user's intended task. | Medium | Low | Confirmed as configuration state: defaults currently select `keras.txt`, `START=1`, so first selected record is bug_id 2. |
| H2 | Joern executable or Joern environment configuration fails early. | Medium | Low | Rejected for current Keras run: `--joern auto` does not require Joern for Keras. Separate note: raw Joern launcher reports missing `greadlink`. |
| H3 | LLM endpoint is still unavailable and causes `main.py --phase algo` to fail. | High | Low | Rejected for this run: one-shot LLM probe returned `MOMO_CURRENT_RUN_OK`; `main.py --phase algo` completed. |
| H4 | Strict target Python/environment creation or target install fails for the current bug record. | Medium | Medium | Partially confirmed and fixed: Python 3.7.3 seed was missing, then Keras rc dependencies were unresolvable; both were addressed. |
| H5 | Recent changes to pyproject target install, bounded combinations, or parallel LLM calls introduced a regression. | Medium | Medium | Rejected by unit tests and one-record progress: `32` tests pass and generation completed for Keras bug 2. |
| H6 | TensorFlow 1.15.0 cannot execute on this Apple Silicon/Rosetta host because the macOS x86_64 wheel requires AVX. | High | Low | Confirmed: V1 probe died with `SIGABRT` and TensorFlow logged `compiled to use AVX instructions, but these aren't available on your machine`. |

## Log Evidence
- Current defaults: `LIB_FILE="keras.txt"`, `LIB_NAME="keras"`, `START=1`, `END=None`; selected records are index `[1:None]`, first selected `bug_id=2`.
- `python3 run_momo_batch.py --preflight-only` passed for the default Keras selection.
- LLM probe against configured `https://www.yunshucode.com/v1`, model `gpt-5.5`, returned `MOMO_CURRENT_RUN_OK`.
- Initial strict environment probe for Keras failed because Python `3.7.3` was unavailable locally and `uv` cannot download `cpython-3.7.3-macos-aarch64-none`.
- Added local x86_64 seed `.momo_runtime/python-seeds/py373-osx64`; `StrictTestEnvironment('3.7.3', ..., provider='auto')` now creates a local exact-version venv successfully.
- Keras requirements contained unavailable historical pins `numpy==1.19.0rc2` and `scipy==1.5.0rc1` in 34 Keras records; both are absent from the current simple indexes.
- Normalization now rewrites those pins to `numpy==1.19.0` and `scipy==1.5.0`; metadata records `rewritten_unavailable_lines`.
- Venv pip is now upgraded to `pip<25`, `setuptools<70`, `wheel<0.43`; this allowed macOS x86_64 wheels to be selected instead of compiling NumPy from source.
- Keras bug 2 rerun progressed through requirements install, `stage_2_function.py`, `main.py --phase algo`, target package install, then reached V1 probe.
- V1 probe failed with `SIGABRT` and TensorFlow message: `The TensorFlow library was compiled to use AVX instructions, but these aren't available on your machine.`
- Manual backend probes showed default/tensorflow backend abort with AVX; `KERAS_BACKEND=numpy` is invalid for Keras 2.2.4; `theano`/`cntk` are not installed and would not match the recorded TensorFlow dependency.

## Verification Conclusion
The current `run_momo_batch.py` entrypoint is structurally working and can reach the analysis stage. The local Keras run is blocked by host CPU compatibility with TensorFlow 1.15.0, not by the entrypoint itself. On this Apple Silicon/Rosetta host, the available TensorFlow 1.15.0 macOS x86_64 wheel requires AVX and aborts during import. Running the default Keras batch therefore requires an Intel x86_64 machine with AVX support, or a different Keras execution strategy/backend that is not equivalent to the recorded TensorFlow environment.
