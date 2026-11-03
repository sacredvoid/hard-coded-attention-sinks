# 11-H-sink-ablation-conservation: H: remove each encoder's sink: does a new one appear, and does it matter?

Branch snapshot of experiment `9ccacc16-ca88-4975-aedc-de311e45bddb`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Mean-ablate each encoder's sink at every layer (means from 100 calib imgs); 500 Imagenette train / 500 val for probe + attention stats.

RESULT (run c33ffef2; d6c0a76e cancelled: targeted wrong register indices):
- Conservation: DINOv2 ablate patch outliers -> more emerge (max 3.04 -> 4.76/img) and the attention sink moves to CLS (top key mass 0.47 -> 0.13, now a prefix token). AIMv2 mean pos-embed at (0,0) -> content outliers double (max 16 -> 34/img), top-key mass 0.48 -> 0.23. SigLIP2 keeps regenerating ~1 outlier/img at the last layer; attention sink mass barely drops (0.33 -> 0.30). CLIP: no regeneration at last layer.
- Function (probe acc): AIMv2 kill positional sink 99.0 -> 91.6 (-7.4), ablate outliers -> 93.4 (-5.6). CLIP 98.8 -> 98.2, DINOv2 99.2 -> 99.8, SigLIP2 99.2 -> 99.0.
- Prefix sinks (reg4 token1, DINOv3 token2, PE CLS): mean-ablation is a no-op (top key mass 0.494 -> 0.493; 0.397 -> 0.397; 0.659 -> 0.659), i.e. the sink token is input-invariant; removing it needs OOD (zero) ablation.
- Massive channels at TRUE sink registers (fixes G2): DINOv2-reg4 token1 top channel 226x patch median; DINOv3 token2 23,417x (most extreme of all encoders).
