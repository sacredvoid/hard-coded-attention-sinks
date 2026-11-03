# 25-V2-small-k-drop-vs-kill: V2: does the sink matter at video-sized budgets, and drop-after vs kill-inside?

Branch snapshot of experiment `a5d4ce3b-e448-4b40-b948-d1780a808ab1`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Same setup as V (LLaVA-OneVision-0.5B, single 384 view, 600 POPE + 500 GQA, paired bootstrap 95% CIs over items).

RESULT (run 7eb05244):
1. Drop-after vs kill-inside (all 729 tokens): dropping the sink's output token = +0.0 pts [-0.3, +0.3]; killing the sink inside the tower (mean pos embed at (21,2)) = -1.4 [-2.5, -0.2]; both = -1.5 [-2.6, -0.3] (POPE -1.5 [-3.0,-0.2], GQA -1.2 [-3.2,+0.8]). The output token is dispensable (agrees with SToP); the sink is structurally needed inside the encoder (disagrees with SToP's 'no structural function').
2. Small budgets, excluding the sink from top-k: k=4 -0.4 [-1.5,+0.7], k=8 +0.3 [-1.2,+1.6], k=16 +1.8 [+0.0,+3.5] (POPE +3.5 [+0.7,+6.3]), k=32 -0.3 [-1.0,+0.5]. No reliable gain: the one-wasted-slot story does not hold even at video-like budgets.
3. Attention-ranked top-k (simplified VisionZip, no merging) is far worse than random at small k: random - topk = +2.7 (k=4), +8.5 (k=8), +15.2 [+12.0,+18.4] (k=16), +5.6 (k=32); POPE topk@16 = 47.0%. Suggests attention ranking fills small budgets with several sink-like tokens, not just the one fixed sink (inference, untested).
