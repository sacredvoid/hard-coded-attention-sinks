# 16-K4-remove-so400m-sink: K4: what happens when you remove the hard-coded sink in SigLIP So400m?

Branch snapshot of experiment `525300d5-643f-434e-bcfc-e5f973bd60df`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

SigLIP v1 So400m/14@384 (sink (21,2)) and SigLIP2 So400m/16@256 (sink (11,0)): clean / swap sink pos embed with grid centre / kill (mean pos embed at sink). Imagewoof probe (1000/1000), 1000 val imgs for stats.

RESULT (run 177ccfa0): swap is free and moves the sink (v1: attn on old patch 0.145 -> 0.001, top-key mass unchanged; acc 95.3 -> 95.2; v2: 95.2 -> 95.2). Kill costs ~2 pts: v1 95.3 -> 93.5, v2 95.2 -> 93.2. After kill the attention sink reappears on other tokens at similar strength (v1 top-key mass at L21 0.126 vs 0.145 clean; v2 0.39 vs 0.36 clean at L21) while in v2 the high-norm patch outliers vanish (1.97 -> 0.02/img), so the new sink is NOT high-norm. v1 So400m's fixed sink is weak (3-14% of attention) and coexists with ~4-6 content outliers/img at 384; v2's is strong (30-40%). Kill block-0 artifact: the edited patch takes 0.38-0.54 of attention at block 0 only.
