# Runtime layout and reuse

The copied `harness/` is the executed control directory. Its source files retain their original hashes. The current remote jobs and environment have not been changed by publishing this repository.

The execution layout is:

```text
RUN_ROOT/
  control/             # this repository's harness/ directory
  repos/FLB/           # this repository's methods/FLB/
  repos/VISTA/         # this repository's methods/VISTA/
  repos/CausalLens/    # this repository's methods/CausalLens/
  repos/CHAIR/         # official scorer dependency pinned in sources.lock.json
  models/              # reused pinned Qwen weights, excluded from Git
  data/coco500/        # reused images/annotations/ground_truth.json, excluded from Git
  envs/qwen/           # locked reusable environment, excluded from Git
  reports/             # frozen audit/configuration files and generated reports
  outputs/             # captions and scores
  state/               # pipeline stage
```

The included Vast environment script targets `/workspace/rebuttal`. Use existing assets and the original environment for current runs. Base dependencies are in `harness/pyproject.toml` and `harness/uv.lock`; the sampled scorer additionally uses the isolated NLTK3.10.3/defusedxml overlay described in `nltk-overlay.txt`. The recorded installed environment is `environment_v2.txt`.

The active parent controller is `harness/scripts/run_dual_decoding.py`, which invokes `run_sampled_baselines.py` for the preserved method checkpoints, with `harness/vast/rebuttal-sampled-baselines.conf`. It follows the normal FLB, VISTA, CausalLens, Regular T1 sequence and reuses completed outputs. It never launches POTION. The source audit script is invoked with `--sampled-baselines`; other legacy code paths are retained only as historical source.

Only set up a new runtime when an experiment is explicitly needed. Do not launch a duplicate of the existing pipeline. Source/audit/frozen configuration files under `harness/reports/` correspond to both CONTROL/reports and RUN_ROOT/reports in the original layout. The CHAIR lexical replay reads the archived reference files from the original runtime; compact proof reports are included here, while the large archived reference dump is left in the original workspace.

Tracking updates use `tracking/update_results.py`. They only copy existing results, render the README and optionally push a commit. No scheduler, cron or monitoring loop is installed.
