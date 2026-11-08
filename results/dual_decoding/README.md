# Greedy and T=1 sampling

Qwen2.5-VL-3B, the same500 images and prompt;512-token cap; NLTK3.10.3 scoring.

## Greedy

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens | Tok/s |
|---|---:|---:|---:|---:|---:|
| Regular | 33.60 | 6.06 | 70.84 | 282.86 | 31.81 |
| FLB | 31.40 | 5.69 | 67.77 | 274.36 | 30.91 |
| VISTA | 31.00 | 5.47 | 68.16 | 316.79 | 20.18 |
| CausalLens | 26.40 | 5.04 | 68.35 | 230.63 | 27.32 |

## Sampling T=1

| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens | Tok/s |
|---|---:|---:|---:|---:|---:|
| Regular | 43.20 | 8.09 | 67.07 | 368.89 | 31.26 |
| FLB | 34.00 | 5.88 | 68.80 | 306.52 | 30.87 |
| VISTA | 41.80 | 8.79 | 63.68 | 349.41 | 20.25 |
| CausalLens | 37.20 | 7.92 | 67.07 | 326.13 | 26.57 |

Recorded preprocessing and timing scopes differ between the two protocols; see comparison.json. All captions and source/run hashes are preserved.
