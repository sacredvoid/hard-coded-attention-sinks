# 01-A-outlier-norms: Outlier-token norms per layer, 7 frozen ViT-L encoders

Branch snapshot of experiment `6f1ea85f-0a60-4f5d-8a0c-b4dd57370cc9`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Claim A from notes/lit-review.md: do SigLIP2, DINOv3, PE-Core, AIMv2 have high-norm outlier patch tokens, and in which layers? CLIP-L/14 and DINOv2-L as references; DINOv2-L+reg4 as negative control. Frozen forward passes on 1000 Imagenette val images at each model's native resolution; residual-stream patch-token L2 norms after every block. Outlier = norm > 4x the image's median patch norm at that layer (also report 3x, 6x).


## Result (2026-11-03)
Patch-token outliers (>4x image median): CLIP ~2.5/img from block 11; DINOv2 ~1/img blocks 17-21; SigLIP2 ~2.3/img from block ~10; AIMv2 fixed position (0,0) at 100-250x from block 0, plus late spread. None in DINOv3 (high norm in one register, ~2200x at block 12) or PE-Core (in CLS, ~22x). DINOv2+reg4 control clean. Report: artifacts/outlier-tokens/outlier-tokens-report.md

