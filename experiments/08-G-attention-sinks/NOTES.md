# 08-G-attention-sinks: G: attention-sink anatomy + AIMv2 (0,0) sink mechanism

Branch snapshot of experiment `0da8cf01-5600-436f-ab5e-8ff2a4599f1f`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Attention mass (patch queries, head-mean) on outliers / prefix tokens / patch (0,0), 300 val images, all 7 encoders. AIMv2 interventions: gray (0,0) patch, roll image 3 patches.

RESULT (run 9aa6e442): every encoder has an attention sink; only its location differs. Patch outliers are sinks: CLIP ~2.5 outliers take 37% of attention (L16), DINOv2 1 outlier 42% (L23), SigLIP2 ~2.3 outliers 50% (L16). Clean encoders move the sink to a prefix token: DINOv2-reg4 register1 46%, DINOv3 register2 40% (L20), PE-Core CLS 64% (L19). AIMv2 patch (0,0) is a sink from layer 0 (43%), unchanged by graying the patch or rolling the image -> positional, not content (BOS-like).
