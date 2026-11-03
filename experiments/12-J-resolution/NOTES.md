# 12-J-resolution: J: do outliers scale with input resolution?

Branch snapshot of experiment `3e1ebb79-3dca-47f5-82b6-a7fc8a0723aa`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Outlier stats (500 val imgs) at 2-3 resolutions per encoder.

RESULT (run adc30e2e): clean encoders stay clean at every resolution (DINOv3 256/512/768, PE 336/448/672, DINOv2-reg4 224/518: max ratio 1.4-2.2, no outliers). Patch-outlier encoders show two scaling regimes for outliers/image: CLIP fixed budget (2.5 @224, 2.8 @336); DINOv2 sublinear (1.0 @224 -> 3.0 @518, 5.3x tokens); SigLIP2 ~token-proportional (2.4 @256, 4.6 @384, 7.9 @512); AIMv2 proportional (16 @224, 27 @336, 49 @448). Emergence layer unchanged by resolution (CLIP 11, DINOv2 17). Caveat: CLIP336/SigLIP2-384/512/AIMv2-336/448 are separate checkpoints; DINOv2/DINOv3/PE use interpolated pos-embed or RoPE.
