# 15-K3-vlm-towers: K3: do VLM vision towers (SigLIP v1/v2 So400m, PaliGemma) carry a fixed sink?

Branch snapshot of experiment `9b9bf33b-dde4-4036-bb5d-e6422f3a9832`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Per-layer outlier/position stats for SigLIP towers used in open VLMs (300 val imgs, native res, T4).

RESULT (run 17093e25): every So400m tower has ONE fixed-position sink (same patch in >90% of images) across most layers:
- SigLIP v1 So400m/14: (12,1) @224 (27/27 layers, from L0); (21,2) @384 (24 layers, from L0).
- SigLIP2 So400m: /14@224 (11,0) [19 layers]; /14@378 (19,0) [18]; /16@384 (16,0) [18]; /16@512 (22,0) [20].
- PaliGemma vision towers keep it after VLM training: PaliGemma1 224 (12,1) [26 layers], PaliGemma2-3B 224 (12,1) [27], PaliGemma2-3B 448 (25,2) [26].
- SigLIP v1 L/16 @256: no fixed sink (content outliers only).
Position is a constant fraction of the grid: v1 row ~0.75-0.78 / col ~0.06; v2 row ~0.67-0.70 / col 0 -> consistent with one learned pos embed interpolated across resolutions.
