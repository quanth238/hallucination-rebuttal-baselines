# Qwen hallucination baseline tracking

FLB, VISTA and CausalLens source snapshots, the Qwen2.5 adapters actually used, and run artifacts for the ICLR rebuttal comparison.

**Snapshot prepared:** 2026-11-08 18:40 Vietnam time (UTC+7). This is updated on request; it is not a live monitor.

## Current sampled runs

The three methods run sequentially: **FLB → VISTA → CausalLens**, using the same supplied 500 images. These target the documented `bacnbt_2` sampling policy. Exact historical output reproduction is not required.

| Method | Status | Images | CHAIR-S ↓ | CHAIR-I ↓ | Object recall ↑ | Avg tokens |
|---|---|---:|---:|---:|---:|---:|
| FLB | 🟠 Running | 74/500 | — | — | — | — |
| VISTA (Qwen adaptation) | ⚪ Queued | 0/500 | — | — | — | — |
| CausalLens | ⚪ Queued | 0/500 | — | — | — | — |

Scores are published after a full 500-image method finishes and passes scoring. Partial caption files are retained as progress snapshots. Only the three requested baselines are scheduled; no new POTION or full Native run is part of this pipeline.

## Completed greedy experiment — supplementary

**These rows use greedy decoding and legacy NLTK 3.9.2 scoring. Keep this table separate from the sampled paper comparison.** The Native row is an existing control.

| Method | Images | CHAIR-S ↓ | CHAIR-I ↓ | Object recall ↑ | Avg tokens | Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Native control | 500 | 33.60 | 6.06 | 70.84 | 282.86 | 31.81 |
| FLB | 500 | 31.40 | 5.69 | 67.77 | 274.36 | 30.91 |
| VISTA (Qwen adaptation) | 500 | 31.00 | 5.47 | 68.16 | 316.79 | 20.18 |
| CausalLens | 500 | 26.40 | 5.04 | 68.35 | 230.63 | 27.32 |

CHAIR and recall are percentages. Recall is auxiliary micro object recall. Tokens/s was measured on an RTX4090 and must stay separate from H100 paper throughput.

In the greedy experiment, VISTA had 19 automated repetition/language flags and 102 cap hits; all captions remain included. FLB retained 309 hallucinated mentions versus Native 309; VISTA had 345 and CausalLens 218. Read recall, lengths and raw counts alongside the hallucination ratios.

## Shared sampled settings

- `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`; BF16, batch 1, KV cache.
- Exact ordered 500 COCO val2014 IDs and image hashes in [`harness/data/coco500.json`](harness/data/coco500.json).
- Prompt: `Please describe this image in detail.`
- Sampling: temperature 1, top-p 1, generation top-k 0, repetition penalty 1, one beam, 512 new tokens.
- Per-image Torch seed: `30260817 + image_id`, reset before image preparation.
- Fast image processor; `max_pixels=1003520`.
- Sampled scoring uses an isolated NLTK 3.10.3 overlay. This exactly reproduces all 500 saved archived Native lexical records and reference metrics.
- Frozen method parameters: FLB gamma=.3/rate=.05/beta=.1; VISTA lambda=.1/alpha=.03/layers29–34; CausalLens lambda=.15/layers11–19.

The supplied cohort was selected post hoc using POT outcomes; it is not an independent random COCO benchmark. FLB/VISTA are source-checked Qwen adaptations, and CausalLens is adapted from its released Qwen2 adapter. The newer POTION 31.4/5.61 row appears in the paper repository report, but its raw run configuration is not committed. Remaining historical defaults are disclosed rather than chased through more reruns.

Exact settings: [`sampled_protocol.json`](harness/reports/protocol_audit_bacnbt2_20261108/sampled_protocol.json).

## Source snapshots and execution code

| Method | Code in this repository | Pinned upstream commit |
|---|---|---|
| FLB | [`methods/FLB/`](methods/FLB/) | [`d32c678c12a8`](https://github.com/jiwooha20/FLB/tree/d32c678c12a8edef428ffe5d9b7de420308ae79b) |
| VISTA | [`methods/VISTA/`](methods/VISTA/) | [`efcf499919e0`](https://github.com/LzVv123456/VISTA/tree/efcf499919e066755e7c33778fbfd864c204329c) |
| CausalLens | [`methods/CausalLens/`](methods/CausalLens/) | [`7d93bf291f8d`](https://github.com/jijy20/CausalLens/tree/7d93bf291f8d3052926c4d058a7b4f8435ded407) |

Upstream source files are unchanged. Generated Python cache files are omitted; licenses and attribution are retained in [`NOTICE.md`](NOTICE.md). The actual Qwen modifications are in [`harness/scripts/flb.py`](harness/scripts/flb.py) and [`harness/scripts/qwen_ports.py`](harness/scripts/qwen_ports.py). The exact executed runner and controller are also in `harness/scripts/`; old source versions remain under `harness/provenance/`.

Both greedy and sampled actual-model 32-token comparisons against released arithmetic passed. Sampled full-cap execution pilots passed for all three methods. These checks establish implementation consistency, not perfect grounding or caption quality.

## Saved artifacts

- [`results/greedy/`](results/greedy/): completed captions, token IDs, run metadata, scores and per-image object matches.
- [`results/sampled/`](results/sampled/): current sampled captions/metadata; completed metrics appear as the pipeline finishes.
- [`tracking/snapshot.json`](tracking/snapshot.json): counts, metrics, status and artifact hashes used to build this README.
- [`harness/`](harness/): source locks, adapters, runners, scorer, 500-image manifest and frozen configurations.
- [`docs/`](docs/): setup/layout notes and the recorded environment.

## Refresh tracking

On the original Mac, run one bounded status/sync and push:

```bash
python3 tracking/update_results.py --refresh --push
```

The local `.tracking.local.json` supplies the original harness/backup paths and is excluded from Git. On another machine, point at an existing backup:

```bash
python3 tracking/update_results.py --source /path/to/baseline-backup --status /path/to/status.json
```

The refresh command copies results and rebuilds this README. It does not launch experiments or install monitoring. Model weights, COCO images, environments, credentials and SSH connection settings are excluded.
