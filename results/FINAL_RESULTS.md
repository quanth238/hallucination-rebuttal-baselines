# Final experiment record

All 8 runs are complete: 500 images per method/protocol, 4000 verified captions. Model: Qwen2.5-VL-3B-Instruct.

## Measured greedy results

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ |
|---|---:|---:|---:|
| Regular | 33.60 | 6.06 | 70.84 |
| FLB | 31.40 | 5.69 | 67.77 |
| VISTA | 31.00 | 5.47 | 68.16 |
| CausalLens | 26.40 | 5.04 | 68.35 |

## Measured sampling T=1 results

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ |
|---|---:|---:|---:|
| Regular | 43.20 | 8.09 | 67.07 |
| FLB | 34.00 | 5.88 | 68.80 |
| VISTA | 41.80 | 8.79 | 63.68 |
| CausalLens | 37.20 | 7.92 | 67.07 |

## Requested paper-reference table

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg. Tok. | Tok/s ↑ |
|---|---:|---:|---:|---:|---:|
| Regular (paper) | 52.60 | 11.00 | 66.80 | 409.50 | 32.38 |
| FLB | 34.00 | 5.88 | 68.80 | 306.52 | 30.87 |
| VISTA | 41.80 | 8.79 | 63.68 | 349.41 | 20.25 |
| CausalLens | 37.20 | 7.92 | 67.07 | 326.13 | 26.57 |
| POT (paper) | 31.40 | 5.60 | 68.16 | 355.10 | 23.55 |

Mixed-source summary. Regular/POT rows quote manuscript Table1; POT denotes POTION. Other rows are new T1 runs. Hardware and timing provenance differ, so throughput is not a matched comparison.

## Preservation

Full captions, token IDs, original run metadata, per-image object matches, scores, configurations, source snapshots and verification hashes are preserved locally and in the GitHub tracking repository.

The measured Regular T1 row is 43.2/8.08963/67.07161; it does not reproduce the recorded manuscript 52.6/11.0/66.8 or archived 53.8/10.88986/68.03069 reference. Keep both measured protocols and the mixed-source presentation table distinct.

Greedy and T1 retain their recorded preprocessing/timing differences. The 500-image subset was selected post hoc using POT outcomes. No POTION generation was included.

User plans to shut down the server. Future review must use these local/GitHub artifacts; do not reconnect or restart experiments without a new request.
