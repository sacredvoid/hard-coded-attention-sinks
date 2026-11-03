# 24-V-llava-ov-pruning: V: the hard-coded sink inside LLaVA-OneVision, and a one-index pruning fix

Branch snapshot of experiment `1f18f3e6-7d15-4d4d-a7ec-1b83119b30ec`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

LLaVA-OneVision-0.5B, single 384 view (no anyres, no image_newline), fp16, T4. VisionZip-style 'dominant' score = attention received at tower block -2 (no contextual-token merging: simplified). 600 POPE + 500 GQA testdev-balanced, greedy, paired bootstrap 95% CIs.

RESULT (run c7367092; eb350950 failed on a transformers-5 module path):
- The fine-tuned tower keeps the hard-coded sink at (21,2), the same patch as timm SigLIP v1 So400m@384: top-norm patch at all 26 blocks in 100% of 100 calibration images. VLM training did not move it.
- VisionZip-style ranking puts the sink at rank 1 in 100% of the 1100 items (so top-k always keeps it).
- Excluding it is worth nothing on single images: overall topk_nosink - topk = +0.0 pts [-0.4, +0.4] at k=64, +0.2 [-0.3, +0.6] at k=144 (POPE -0.2 / +0.3; GQA +0.2 / +0.0). One wasted slot of 64 does not show.
- High-norm filter at 4x median never fired: 0 tokens >4x at the feature layer (sink is top-norm but <4x there), so 'topk_nohinorm' == topk. Not a valid SinkPruner stand-in.
- Pruning itself: full 71.5; topk@64 62.1, random@64 62.9; topk@144 66.6, random@144 66.3 (simplified VisionZip is not better than random here; GQA random >= topk by ~2 pts, CI spans 0).
