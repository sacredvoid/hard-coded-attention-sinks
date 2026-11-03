# 03-C-global-info: Claim C: do outlier tokens carry global information?

Branch snapshot of experiment `09cbceae-834a-4e33-bd1d-3d422eb43012`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Linear probes (Imagenette, 10 classes) on frozen features at the outlier layers of CLIP, DINOv2, SigLIP 2, AIMv2: top-norm outlier patch token vs random normal patch vs mean-pooled patches vs CLS.

RESULT (run f91d2724): outlier >> normal patch in all 4, so outliers carry image-level info. CLIP L13 outlier 92.7 / normal 68.2 / mean 96.3; DINOv2 L19 97.8/76.1/98.3; SigLIP2 L15 97.1/80.4/98.5; AIMv2 L12 (fixed (0,0) sink) 85.2/67.2/95.9. Shuffled-label control ~10%. Caveat: Imagenette near ceiling -> C2 on Imagewoof.
