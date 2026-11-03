# 20-A1-aimv2-collapse: A1: why does AIMv2 collapse without its positional sink?

Branch snapshot of experiment `6e136bdc-e679-4199-96e8-54120de717a8`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

AIMv2 L/14, Imagewoof probe (1000/1000, seed 0, features = mean of patches excluding token 0), paired bootstrap vs clean.

RESULT (run bfdf36d2): nothing rescues it; damage happens inside block 0.
- clean 95.0; kill 65.0 (-30.0 [-33.1,-27.0]).
- Dose-response is a cliff: pos embed blended 25/50/75% toward the mean costs -0.2/-0.1/-0.1 (all CIs include ~0); 100% costs -30.
- Patching the clean token 0 into the killed model at block 0/1/2/4/8/12 output: 68.3/66.3/67.7/67.0/66.9/65.9 (no rescue even right after block 0 -> the other patches are corrupted when they read the altered sink during block 0's attention).
- Restoring only token 0's top-5 massive channels at block 1/4: 64.1/63.9 (no rescue).
- Divergence (32 val imgs): cosine of clean vs killed mean patch token 0.63 (block 0) -> 0.17 (1) -> ~0.3 (2-8) -> 0.02-0.1 (10-20) -> 0.71 (23). Token-0 norm/median: clean 112-248x, killed 1.7-2.2x at blocks 0-2, 23-70x later (the killed model rebuilds a weaker sink by block 3).
