# 28-D-VLM-register-in-llava: D-VLM: does the constant-register fix work inside LLaVA-OneVision?

Branch snapshot of experiment `e577f857-047d-43e2-a8f4-54debc4b4c55`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

LLaVA-OneVision-0.5B tower: kill the (21,2) pos-embed sink; append a constant register (mean clean layer-0 input at the sink over 100 Imagenette train images) inside the tower, dropped before the projector; all 729 patch tokens to the LLM. 1100 POPE/GQA items, paired bootstrap vs full.

RESULT (run 3c6e8f4c): partial recovery only. Overall: kill -1.4 [-2.5,-0.2]; kill+reg -1.0 [-2.2,+0.2]; kill+zero -1.4 [-2.5,-0.1]; clean+reg -0.3 [-0.8,+0.3]. POPE: kill -1.5, kill+reg -1.3; GQA: kill -1.2, kill+reg -0.6. The register recovers ~30% of the gap in the fine-tuned VLM tower, vs 67-100% in the bare encoders (run D). Sink-patch norm ratio at the last block: 3.5x clean, ~1.0x when killed (with or without the register).
