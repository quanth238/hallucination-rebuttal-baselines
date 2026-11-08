# Greedy and T=1 sampling

The comparison covers Regular, FLB, VISTA and CausalLens on the supplied 500 COCO images, using the same prompt and a 512-token cap. All current scores use NLTK 3.10.3.

| Protocol | Token selection | Generation settings |
|---|---|---|
| Greedy | `do_sample=False` | One beam, repetition penalty 1 |
| Sampling T=1 | `do_sample=True` | Temperature 1, top-p 1, top-k 0, repetition penalty 1 |

The temperature value alone does not determine whether decoding is greedy. The sampling flag controls token selection.

## Greedy results

All four methods have 500 complete captions. Their original captions, run metadata and scores remain unchanged in `results/greedy/`. CPU-only rescoring with NLTK 3.10.3 is saved separately in `results/greedy_verified/`. The README uses the rescored values. No repeated greedy generation was needed.

## Sampling runs

The normal sequence is FLB, VISTA, CausalLens, then Regular T=1. FLB is complete; VISTA resumes from its saved checkpoint. All four sampled methods use the same RTX4090 and sampled settings, with per-image Torch seed30260817+image_id.

Existing checkpoints and source/run hashes are preserved. `run_dual_decoding.py` runs the stages sequentially and builds the final two-protocol report. The Regular reference check is produced after Regular finishes; it does not change the queue or force a target numeric score.

## Reading the comparison

These are two recorded evaluation protocols. The greedy runs use the slow image processor; the sampled runs use the fast processor. All recorded processed images fit below the sampled pixel cap, but resizing arithmetic can differ. Greedy timing excludes image preparation; sampled timing includes it. The results therefore are not a strict decoder-only ablation. Keep the actual configuration and timing scope when interpreting differences.

The supplied 500-image cohort was selected post hoc using POT outcomes; the manifest retains that provenance. Keep every caption, including quality flags. Detailed settings and hashes are retained in each run.json and in the verification reports.

Archived repository references remain under `results/reference/`. Historical H100 throughput is not combined with the new RTX4090 timing. The new sampled Regular row is the control for the new sampled method rows; it explicitly uses repetition penalty 1. No POTION generation is included.

After Regular scoring finishes, `results/sampled/regular/reference_check.json` records measured differences from both the manuscript and archived branch reference rows. It does not adjust parameters to force a numeric match.
