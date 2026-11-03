# 10-G2-pos-embed-swap-massive-channels: G2: is AIMv2's sink set by the (0,0) position embedding? + massive channels

Branch snapshot of experiment `3bf1cba2-a9d7-47a1-a372-aaab772bcf65`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

AIMv2 pos-embed edits + massive channels at sink tokens (200 val imgs).

RESULT (run 8048b620; first attempt 37d772de killed during channel pass, but its pos-embed part matched): swapping AIMv2's (0,0) pos embed with (8,8) or (15,15) moves the top-norm token AND the attention sink to the target in 100% of images at all 24 layers, with the same attention mass (0.43 at L0). Replacing (0,0)'s pos embed with the mean pos embed keeps (0,0) top-norm but kills most of the sink (attn 0.43 -> 0.002 at L0, ~0.03 mid layers). Sink is written by one learned position embedding.
Massive channels: in all 7 encoders the sink token's top-1 channel is the same channel in 100% of images; top-5 channels = 80-98% of its squared norm. Top channel vs patch median: AIMv2 15527x (L4), CLIP 1322x, SigLIP2 1213x, PE CLS 1108x, DINOv2 697x, DINOv3 register 69x, DINOv2-reg4 register 35x.

CORRECTION: for dinov2_l14_reg4 and dinov3_l16 the channel pass used prefix token idx 2 and 3, but run G shows the sinks are tokens 1 and 2. Their 35x / 69x numbers describe NON-sink registers. Re-measured at the true sink in node H.

