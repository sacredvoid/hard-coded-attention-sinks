# 13-K-model-scale: K: does each family keep its sink location across model scale?

Branch snapshot of experiment `fa4ac65b-a148-4ec7-a51e-0daf29cad6ee`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Per-layer outlier/position/prefix-norm stats for 18 other-size checkpoints (300 val imgs, native res, A10G).

RESULT (run ce899d36): sink location is mostly a family trait, with scale effects.
- Fixed (0,0) positional sink at all 24 layers in 100% of imgs: AIMv2 H, 1B, 3B and L-distilled.
- NEW: SigLIP2 So400m has ONE fixed-position sink at (11,0) in 100% of imgs (L12 ratio 184x) and SigLIP2 g at (0,13) from L4 (15-18x); SigLIP2 B (and L from parent) have content-dependent outliers instead.
- Content outliers: CLIP B/16 OpenAI (~10/img), DINOv2 B (~4/img) and g (~26/img from L19); DINOv2 S essentially none (max 3.9x). PE-Core G (no CLS token in timm) HAS patch outliers (~12/img from L31) while PE B/L (with CLS) are clean.
- Prefix sinks: DINOv3 S/B/H+ clean, one register high-norm (B tok1 13.6x, H+ tok2 17.4x); DINOv2 g reg4 clean, tok3 11.9x; PE B CLS 7.5x.
- Anomaly: CLIP H/14 LAION-2B shows no high-norm token at all (max ratio 1.78, CLS 0.99x).
Note: the one-line JSON summary got truncated at 64 KiB in the log; numbers taken from the per-layer tables.
