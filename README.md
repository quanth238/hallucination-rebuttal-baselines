# Hallucination baselines

Regular, FLB, VISTA and CausalLens with Qwen2.5-VL-3B-Instruct on the same 500 COCO images.

Results snapshot: 2026-11-09 05:49 (UTC+7).

Both protocols use a 512-token cap and NLTK 3.10.3 scoring.

## Greedy

`do_sample=False`, 500 images per method.

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens | Tok/s |
|---|---:|---:|---:|---:|---:|
| Regular | 33.60 | 6.06 | 70.84 | 282.86 | 31.81 |
| FLB | 31.40 | 5.69 | 67.77 | 274.36 | 30.91 |
| VISTA | 31.00 | 5.47 | 68.16 | 316.79 | 20.18 |
| CausalLens | 26.40 | 5.04 | 68.35 | 230.63 | 27.32 |

## Sampling T=1

`do_sample=True`, temperature 1, top-p 1, top-k 0; 500 images per method.

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens | Tok/s |
|---|---:|---:|---:|---:|---:|
| Regular | 43.20 | 8.09 | 67.07 | 368.89 | 31.26 |
| FLB | 34.00 | 5.88 | 68.80 | 306.52 | 30.87 |
| VISTA | 41.80 | 8.79 | 63.68 | 349.41 | 20.25 |
| CausalLens | 37.20 | 7.92 | 67.07 | 326.13 | 26.57 |

CHAIR and recall are percentages. Timing uses an RTX4090. Recorded preprocessing and timing scopes are described in the [evaluation notes](docs/EVALUATION.md).

## Code

| Method | Source | Upstream |
|---|---|---|
| FLB | [methods/FLB](methods/FLB/) | [Original repository](https://github.com/jiwooha20/FLB/tree/d32c678c12a8edef428ffe5d9b7de420308ae79b) |
| VISTA | [methods/VISTA](methods/VISTA/) | [Original repository](https://github.com/LzVv123456/VISTA/tree/efcf499919e066755e7c33778fbfd864c204329c) |
| CausalLens | [methods/CausalLens](methods/CausalLens/) | [Original repository](https://github.com/jijy20/CausalLens/tree/7d93bf291f8d3052926c4d058a7b4f8435ded407) |

[Run files and scores](results/) · [Experiment code](harness/) · [Evaluation notes](docs/EVALUATION.md) · [Setup](docs/RUNNING.md)

[Verified greedy scores](results/greedy_verified/) retain the original [captions and run records](results/greedy/). Archived repository reference values remain in [results/reference/](results/reference/).

The supplied 500-image subset was selected using POT outcomes; selection details are retained in the [image manifest](harness/data/coco500.json).

## Update results

```bash
python3 tracking/update_results.py --refresh --push
```

Upstream source versions and licenses are retained in [SOURCES.json](methods/SOURCES.json) and [NOTICE.md](NOTICE.md).
