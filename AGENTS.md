# Baseline tracking repository

Read README.md, tracking/snapshot.json, methods/SOURCES.json and the sampled protocol before acting. This repository tracks the existing FLB, VISTA and CausalLens experiments; it does not authorize launching new experiments.

Keep the current sampled runs and frozen method settings. Exact reproduction of historical paper outputs is not required. Do not run POTION, repeat completed greedy generation, tune on the500 test results, or install continuous monitoring. A new Regular sampled control is explicitly authorized. Preserve the prior greedy outputs as a separate supplementary table.

When the user requests a progress/result update, use the original Mac harness through `python3 tracking/update_results.py --refresh --push` if `.tracking.local.json` is available. This performs one status check/sync and pushes only the managed README/results/snapshot. No background watcher is needed. Keep SSH details and local paths in the ignored local config.

The upstream method snapshots retain their own copyright/license notices. Execution code and exact executed runners are in harness/. Retain source and artifact hashes; record an actual bug fix separately rather than rewriting historic provenance. Keep detailed provenance and quality notes in the run metadata. Keep the main README brief, use plain method names, and show completed results only; do not add running/queued tables or adaptation labels.

The user explicitly requested a public repository and a short README. Preserve that preference when regenerating tracking updates.

Show separate Greedy and Sampling T1 tables. Use NLTK3.10.3 rescored greedy values from results/greedy_verified/ and preserve original artifacts in results/greedy/. Keep archived references separate from the new measured Regular controls.

Latest user direction: keep BOTH greedy and sampling T=1 comparisons for Regular, FLB, VISTA and CausalLens. Greedy generation/scoring is complete; resume the preserved sampled FLB checkpoint, then VISTA/CausalLens, followed by a new Regular T1 control. No POTION run. Keep separate completed-results tables and retain the actual preprocessing/timing metadata. The parent execution controller is run_dual_decoding.py.

Latest explicit instruction: do not prioritize Regular. Follow the normal sequence: completed FLB, resume VISTA, then CausalLens, then Regular T1. Reuse all checkpoints and completed greedy results. Do not reorder without an explicit request. The Regular reference check is produced after its500 images are scored.

Completed2026-11-09: all8 runs have500 images and4000 verified captions total. Preserve the final tables, reference mismatch and tracking/final_verification.json. Do not restart inference or poll without a new request. Vast lifecycle changes require a user instruction.
