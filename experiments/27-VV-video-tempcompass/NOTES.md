# 27-VV-video-tempcompass: VV: does dropping the per-frame sink help video QA (TempCompass)?

Branch snapshot of experiment `8f1f6148-b3d4-44aa-85e3-43da3df1d788`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

LLaVA-OneVision-0.5B on TempCompass multi-choice (1580 Qs, 410 videos, 16 uniform frames), HF-style per-frame pooling (196 tokens/frame + newline). Pooled sink index calibrated from pooled encoder attention = 155 = (11,1), the index SToP (2604.20937) reports. Paired bootstrap 95% CIs.

RESULT (run a66c6957): the per-frame sink is a non-issue for this video QA setup.
- full 53.2; drop the pooled sink in every frame +0.4 [-0.3,+1.1] (action +1.8 [0.0,+3.6], speed -0.9, others within +-0.7); kill the sink inside the tower +0.0 [-0.9,+0.9] (no cost on video, unlike -1.4 on images in V2).
- Per-frame top-k by pooled encoder attention: k=8 46.7, excluding the sink -0.1 [-1.3,+1.1]; k=16 48.4, excluding the sink +0.3 [-0.6,+1.3]. Random-k beats top-k: +3.0 [+1.1,+4.8] (k=8), +1.5 [-0.3,+3.3] (k=16).
SToP's gain from suppressing sinks is not reproduced here (0.5B, TempCompass MCQ, single-index exclusion; SToP scores many sink tokens and evaluates other benchmarks).
