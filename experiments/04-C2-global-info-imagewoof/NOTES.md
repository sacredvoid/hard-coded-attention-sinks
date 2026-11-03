# 04-C2-global-info-imagewoof: C2: global info on fine-grained Imagewoof + where clean encoders keep it

Branch snapshot of experiment `334b07b5-ee9a-4baa-be60-763e37829ef8`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Same probes as C on Imagewoof (10 dog breeds), plus prefix-token probes for clean encoders (prefix0=CLS, prefix1-4=registers).

RESULT (run 6afabb98): outlier >> normal widens on harder data. CLIP L13 69.5 vs 32.3 (mean 83.9); DINOv2 L19 86.8 vs 39.4 (mean 89.8, CLS 92.8); SigLIP2 L15 84.7 vs 54.6 (mean 93.4); AIMv2 (0,0) sink L12 58.8 vs 31.2 (mean 79.0), L23 92.2 vs 58.3. Clean encoders: sink registers are NOT the most informative: DINOv3 sink token 2 = 57.2 (L12) / 81.2 (L23) vs other registers 70-74 / 96; DINOv2-reg4 sink token 1 = 92.3 vs others 84-95 (L19). PE CLS (sink) 91.4 vs mean 94.7 at L23. Shuffled-label control 8-12%.
