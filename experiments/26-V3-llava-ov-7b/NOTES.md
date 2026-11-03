# 26-V3-llava-ov-7b: V3: replicate drop-after vs kill-inside on LLaVA-OneVision 7B

Branch snapshot of experiment `c3cfe9da-76ff-48a0-8ba6-d1831b0c5887`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

V2 replicated on llava-onevision-qwen2-7b-ov-hf (A10G), same 1100 POPE/GQA items, paired bootstrap 95% CIs.

RESULT (run 5365c0a1): replicates V/V2. 7B tower: sink at (21,2), top-norm at every block in 100% of calibration images; attention rank 1 in 100% of items. Overall: full 74.8; drop sink output token +0.1 [+0.0, +0.4]; kill inside encoder -1.2 [-2.5, +0.0] (POPE -1.0 [-2.3,+0.3], GQA -1.4 [-3.4,+0.6]); kill+drop vs drop -1.3 [-2.5, -0.2]. Attention-ranked top-16 vs random-16: random better by +19.5 [+16.4, +22.7] (POPE topk@16 47.2%, random 74.3%). Same direction and size as 0.5B (V2: kill -1.4 [-2.5,-0.2], random-topk@16 +15.2).
