# Evaluation settings

The primary comparison uses the documented sampling policy from `ngquin102/Hallucination-VLM`, branch `bacnbt_2`, commit 458002a12c7158050ee3b0822ed5eb81c9e89c0b: Qwen2.5-VL-3B, the supplied 500 images and prompt, T=1/top-p=1/top-k=0, per-image Torch seed 30260817+image_id, and a 512-token quality cap. New method rows use repetition penalty 1.0 as specified by the POT report. CHAIR uses the repository utility with NLTK 3.10.3; archived parsing and reference metrics reproduce exactly.

The Qwen reference row is imported from the branch's archived Native output and first 512-token scoring; its provenance and raw-output hash are in `results/reference/bacnbt_2_qwen_native.json`. It is not a newly generated 4090 control. Native source inherits checkpoint generation defaults (1.05 repetition penalty on the current pinned checkpoint), while the POT report specifies 1.0; historical model revision/default metadata were not fully recorded. These remaining differences are retained in metadata rather than addressed through extra matching runs.

The older greedy results remain under `results/greedy/`; they are supplementary and are not the primary sampled comparison. New results are under `results/sampled/`. The original greedy scores and source/run hashes are preserved.

The repository reference timing came from a 1024-cap H100 run, while the new runs measure direct 512 generation on an RTX4090. Do not mix these throughput values or infer a hardware-independent speed advantage.

Use measured outputs without tuning parameters to reproduce a target score. Keep all 500 images and all captions. The supplied cohort was selected post hoc using POT outcomes; the manifest retains that provenance. Exact historical output reproduction is not required.
