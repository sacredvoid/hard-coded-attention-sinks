# 07-F-perturbation-fragility: F: are outlier tokens brittle under tiny perturbations?

Branch snapshot of experiment `c1a1ade1-f57d-4a10-895a-dab96f15f78b`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Outlier-position stability under perturbation (300 val imgs) + PGD create/remove (100 imgs, eps 1/255 and 4/255, 10 steps). Metric 'kept' = share of clean outliers with a perturbed outlier within 1 patch.

RESULT (run 79a25608; first attempt 511b74b3 crashed on an arg bug): outlier COUNT is conserved but POSITIONS are fragile. Gaussian noise sigma=1/255 keeps only 74% (CLIP L13), 78% (DINOv2 L19), 91% (SigLIP2 L15); sigma=4/255: 58/57/76%. Count unchanged (CLIP 2.51->2.51). JPEG q75 keeps >=98%. Flip keeps (mirrored) 52% CLIP, 27% DINOv2, 54% SigLIP2 -> positions not content-equivariant. AIMv2 stable (noise1 98%, noise16 94%). Clean encoders (reg4, DINOv3, PE): zero outliers under any perturbation. PGD on the norm ratio failed: create success <=8%, remove 0-1% (weak attack, 10 steps; read as 'not easily steerable', not 'robust').
