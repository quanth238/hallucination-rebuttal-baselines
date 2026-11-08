# Hallucination baselines

FLB, VISTA and CausalLens experiments with Qwen2.5-VL-3B-Instruct on the same 500 COCO images as the `bacnbt_2` evaluation.

Results snapshot: 2026-11-08 18:59 (UTC+7).

Sampling: temperature 1, top-p 1, top-k 0; 512-token cap; per-image seeds; NLTK 3.10.3 scoring.

## Repository reference

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens |
|---|---:|---:|---:|---:|
| Qwen2.5-VL baseline | 53.80 | 10.89 | 68.03 | 408.49 |

Reference reused from [`bacnbt_2`](https://github.com/ngquin102/Hallucination-VLM/blob/458002a12c7158050ee3b0822ed5eb81c9e89c0b/QOT_steer/reports/latest_full_benchmarks_512/README.md#L37-L41); [provenance](results/reference/bacnbt_2_qwen_native.json).

Completed FLB, VISTA and CausalLens scores will be added after the sampled runs finish. [Run files](results/sampled/) · [Progress snapshot](tracking/snapshot.json).

CHAIR and recall are percentages. Recall is micro object recall. New timings use an RTX4090; reference timing is kept separate.

## Code

| Method | Source | Upstream |
|---|---|---|
| FLB | [methods/FLB](methods/FLB/) | [Original repository](https://github.com/jiwooha20/FLB/tree/d32c678c12a8edef428ffe5d9b7de420308ae79b) |
| VISTA | [methods/VISTA](methods/VISTA/) | [Original repository](https://github.com/LzVv123456/VISTA/tree/efcf499919e066755e7c33778fbfd864c204329c) |
| CausalLens | [methods/CausalLens](methods/CausalLens/) | [Original repository](https://github.com/jijy20/CausalLens/tree/7d93bf291f8d3052926c4d058a7b4f8435ded407) |

[Run files and scores](results/) · [Experiment code](harness/) · [Evaluation notes](docs/EVALUATION.md) · [Setup](docs/RUNNING.md)

The earlier [greedy results](results/greedy/) are a separate supplementary experiment.

The supplied 500-image subset was selected using POT outcomes; selection details are retained in the [image manifest](harness/data/coco500.json).

## Update results

```bash
python3 tracking/update_results.py --refresh --push
```

Upstream source versions and licenses are retained in [SOURCES.json](methods/SOURCES.json) and [NOTICE.md](NOTICE.md).
