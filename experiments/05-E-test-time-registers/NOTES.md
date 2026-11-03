# 05-E-test-time-registers: Claim E: do test-time registers transfer?

Branch snapshot of experiment `50390abe-6fe4-4c27-897b-35dee83b1a31`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Test-time registers (Jiang et al. 2506.08010) applied unchanged: register neurons = top-K by mean activation at outlier positions (200 Imagenette calib imgs, layers <= DET), shifted onto one appended zero token. Measured on 1000 val imgs; Imagenette linear probe on mean-pooled patches.

RESULT (run cb3305a8; dd6e5a62 crashed on a class-split bug): transfer is uneven; probe accuracy unchanged everywhere (99.0-99.8%).
- CLIP L: full transfer. Last-layer outliers 2.26 -> 0.78 (K=10) -> 0 (K=20); K=50 removes them at all layers.
- SigLIP2 L: works at K>=20: last-layer 2.36 -> 0.33 (K=20) -> 0 (K=50); one mid-layer outlier/img remains.
- DINOv2 L: unstable: peak-layer outliers 3.13 -> 1.16/1.76/3.56/0.73 for K=5/10/20/50 (non-monotonic).
- AIMv2 L: fails on content outliers: 16 -> ~11/img even at K=50 (max ratio 25 -> 9.5; register absorbs 21x).
- Adding the unused zero token alone (reg_only) RAISES outliers in DINOv2 (3.1 -> 5.4) and AIMv2 (16 -> 19.4).
