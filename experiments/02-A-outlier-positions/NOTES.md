# 02-A-outlier-positions: Outlier positions: fixed-position vs content-dependent

Branch snapshot of experiment `5824e0ed-1f38-45d0-b5d7-4b50aa1ec3c8`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Builds on the baseline's finding that AIMv2 and SigLIP2 have exactly one >4x token per image from layer 0 while CLIP/DINOv2 outliers appear mid-network. Same measurement; additionally record, per layer, the patch positions of >4x tokens and how concentrated they are (share at the single most common position, number of distinct positions). If an encoder's outliers sit at one position in every image, they are a positional artifact, not Darcet-style content-dependent outliers. Also: dump summary.json to stdout (Modal discards outputs/), keep prefix norms float32 (float16 overflowed), drop the npz dump.


## Result (2026-11-03)
Patch-token outliers (>4x image median): CLIP ~2.5/img from block 11; DINOv2 ~1/img blocks 17-21; SigLIP2 ~2.3/img from block ~10; AIMv2 fixed position (0,0) at 100-250x from block 0, plus late spread. None in DINOv3 (high norm in one register, ~2200x at block 12) or PE-Core (in CLS, ~22x). DINOv2+reg4 control clean. Report: artifacts/outlier-tokens/outlier-tokens-report.md

