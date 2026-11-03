# 09-G-rep-replication: G-rep: attention-sink anatomy replicated on Imagewoof (seed 1)

Branch snapshot of experiment `348d2ba2-38e4-46d1-aff5-39f8a7c311e1`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Replication of G on 300 Imagewoof val images, seed 1 (G: Imagenette, seed 0).

RESULT (run 7157ab01): replicates. Peak top-key mass (block >2), G -> rep: CLIP 0.281 -> 0.306 (L22), DINOv2 0.471 -> 0.454 (L21), DINOv2-reg4 0.495 -> 0.494 (token 1, L21), SigLIP2 0.336 -> 0.355 (L18/17), DINOv3 0.399 -> 0.402 (token 2, L20), PE-L 0.658 -> 0.684 (CLS, L21), AIMv2 0.479 -> 0.544 ((0,0), L19). Same sink token in every encoder; outliers/img at peak within 0.13. AIMv2 gray00/roll3 again leave the (0,0) sink unchanged (L0 0.426/0.423 vs 0.422).
