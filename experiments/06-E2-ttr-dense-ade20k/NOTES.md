# 06-E2-ttr-dense-ade20k: E2: do test-time registers help dense prediction (ADE20K linear seg)?

Branch snapshot of experiment `e198faba-d326-40b3-85c5-523573b703d2`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

ADE20K linear dense probe (1000 train / 500 val, patch-grid labels, final-layer patch tokens) with E's test-time registers at K = 10/20/50.

RESULT (run 2b7e7e71): test-time registers do not improve dense prediction (mIoU changes within about 1 pt): CLIP 18.61 -> 18.53/18.12/17.66 (slightly worse), DINOv2 31.51 -> 31.65/31.69/31.87, SigLIP2 22.84 -> 22.68/22.83/22.84, AIMv2 18.81 -> 18.93/18.93/18.98. Outlier patches really do lose local semantics (patch accuracy at outlier vs normal positions, base): CLIP 46.1 vs 58.6, SigLIP2 20.1 vs 60.6, AIMv2 37.8 vs 57.8 (DINOv2 has no final-layer outliers). They are too few (<1-3% of patches) to move mIoU.
