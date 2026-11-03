# 19-W-weight-only-detector: W: can you find the hard-coded sink from the weights alone?

Branch snapshot of experiment `fbed6dd7-f588-455f-b213-b7b1f6674540`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Weight-only detector: z-score pos-embed rows by norm, distance from mean row, and residual from 4-neighbour mean; compare the top row with the empirically found sink. Modal cpu-large.

RESULT (run 5fe2568c): works for some families only.
- AIMv2 L/L-dist/H/1B: sink row (0,0) is a huge outlier (norm z 15.7-15.8, next row 1.7-2.4; neighbour-residual z 14.2-14.4 vs 4.5). Clean detection.
- SigLIP v1 So400m 224/384 and PaliGemma 1 / 2-3B 224/448 towers: sink row is the top row on all three scores but with a thin margin (z 9.9-16.6 vs next 7.1-13.4).
- SigLIP2 So400m (224/256/378/384/512) and g: FAILS. Sink row unremarkable (norm z -0.2..1.8, rank 6-752); top rows are other left-edge/top-edge rows ((3,0), (5,0), (6,0), (2,15)). Yet swapping the sink row moves the sink (K2), so SigLIP2 encodes it in a direction the norm does not show.
- Negatives also have outlier rows: SigLIP2 L (13,15) z=12.6 (matches the weak early-block fixed token at (13,15) seen in claim A), SigLIP v1 L (3,15) z=9.5, SigLIP2 B (2,0) z=8.7; CLIP L OpenAI max 3.1. Outlier pos-embed rows predict early fixed high-norm tokens, not necessarily the late attention sink.
