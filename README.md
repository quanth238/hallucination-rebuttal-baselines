# Hallucination baselines

FLB, VISTA and CausalLens experiments with Qwen2.5-VL-3B-Instruct on 500 COCO images.

Results snapshot: 2026-11-08 18:40 (UTC+7).

## Results

Greedy decoding, 500 images per method.

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens | Tok/s |
|---|---:|---:|---:|---:|---:|
| Native | 33.60 | 6.06 | 70.84 | 282.86 | 31.81 |
| FLB | 31.40 | 5.69 | 67.77 | 274.36 | 30.91 |
| VISTA | 31.00 | 5.47 | 68.16 | 316.79 | 20.18 |
| CausalLens | 26.40 | 5.04 | 68.35 | 230.63 | 27.32 |

CHAIR and recall are percentages. Recall is micro object recall. Timing uses an RTX4090.

## Code

| Method | Source | Upstream |
|---|---|---|
| FLB | [methods/FLB](methods/FLB/) | [Original repository](https://github.com/jiwooha20/FLB/tree/d32c678c12a8edef428ffe5d9b7de420308ae79b) |
| VISTA | [methods/VISTA](methods/VISTA/) | [Original repository](https://github.com/LzVv123456/VISTA/tree/efcf499919e066755e7c33778fbfd864c204329c) |
| CausalLens | [methods/CausalLens](methods/CausalLens/) | [Original repository](https://github.com/jijy20/CausalLens/tree/7d93bf291f8d3052926c4d058a7b4f8435ded407) |

[Run files and scores](results/) · [Experiment code](harness/) · [Settings](harness/reports/protocol_audit_bacnbt2_20261108/sampled_protocol.json) · [Setup](docs/RUNNING.md)

The supplied 500-image subset was selected using POT outcomes; selection details are retained in the [image manifest](harness/data/coco500.json).

## Update results

```bash
python3 tracking/update_results.py --refresh --push
```

Upstream source versions and licenses are retained in [SOURCES.json](methods/SOURCES.json) and [NOTICE.md](NOTICE.md).
