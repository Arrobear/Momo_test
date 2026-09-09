# Debug Session: luigi-cut-stall
- **Status**: [OPEN]
- **Issue**: Terminal appears stuck after Luigi `check_condition_filter` finishes and `luigi_cut_combinations_0.json` is created.
- **Debug Server**: not started; first pass uses process state and artifact timestamps.
- **Log File**: pending

## Reproduction Steps
1. Run the current Luigi batch from `run_momo_batch.py`.
2. Observe terminal output around:
   `check_condition_filter`
   `[Created] .../arg_combinations/luigi_cut_combinations_0.json`
3. Terminal appears to stop printing after that point.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| H1 | `cut_combinations` is still running and has no progress logging after creating the output file. | Medium | Low | Rejected: the process had moved past `check_condition_filter`; the real issue was huge generated artifacts. |
| H2 | `generate_api_boundary` has started and is waiting on parallel LLM calls before enough output is visible. | High | Low | Rejected for the original stall: no boundary file existed; it had not reached boundary generation. |
| H3 | One or more LLM API calls are retrying or blocked, so the process is alive but not advancing. | Medium | Low | Rejected for the stall point: CPU and file evidence pointed to local path-space explosion, not LLM retries. |
| H4 | The child `main.py --phase algo` process is stuck or deadlocked while parent `run_momo_batch.py` waits. | Medium | Medium | Reframed: child was alive, but processing/writing a pathological path space rather than deadlocked. |
| H5 | Static path enumeration generated too many paths for `luigi.scheduler.Scheduler.add_task`, causing downstream JSON explosion. | High | Low | Confirmed: `luigi_arg_space_0.json` was 2.3GB with 1,658,883 paths; `luigi_cut_combinations_0.json` grew to 9.3GB. |

## Log Evidence
- Active process evidence before interruption: `main.py --phase algo --k 1` was still alive and using CPU while `run_momo_batch.py` waited.
- Generated artifact sizes at stall:
  - `documentation/arg_space/luigi_arg_space_0.json`: 2.3GB
  - `documentation/arg_combinations/luigi_cut_combinations_0.json`: 9.3GB
  - original `luigi_combinations_0.json`: 14KB
  - `error_luigi_combinations.json`: 159B
- JSON metadata before fix showed `luigi.scheduler.Scheduler.add_task` had `1,658,883` static paths.
- `Scheduler.add_task` had only `212` parameter combinations and `2` error combinations, so the explosion came from path enumeration, not parameter combinations.
- Added representative static-path selection with `MOMO_MAX_STATIC_PATHS_PER_API` defaulting to `64`.
- Post-fix targeted verification with `MOMO_LIB_NAME=luigi MOMO_LIB_GITNAME=luigi MOMO_MAX_STATIC_PATHS_PER_API=64`:
  - `stage_2_function.py` wrote `luigi_arg_space_0.json` as 284B with 1 path for the current API.
  - `cut_combinations()` wrote `luigi_cut_combinations_0.json` as 859B with 1 path group and 2 combinations.
- A later full run with 64 retained paths still produced `5782` boundary LLM tasks because each path kept up to 93 parameter combinations.
- Added `MOMO_MAX_BOUNDARY_COMBINATIONS_PER_PATH` with default `4`.
- Post-fix targeted verification for the 64-path Luigi artifact:
  - `luigi_cut_combinations_0.json` is 62KB.
  - path groups: `64`.
  - total boundary LLM tasks: `256`.
  - max combinations per path: `4`.
- Unit tests: `python3 -m unittest test_run_momo_batch.py` ran 35 tests and passed.

## Verification Conclusion
The apparent hang was caused first by static path explosion for `luigi.scheduler.Scheduler.add_task`, then by too many boundary-generation combinations per retained path. Path-space capping plus per-path boundary-combination capping now keep both `arg_space` and `cut_combinations` bounded before boundary/test generation.
