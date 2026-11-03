# 17-K4-CI-kill-cost-cis: K4-CI: confidence intervals for the cost of killing a positional sink

Branch snapshot of experiment `4d58a7b9-ddb2-4b1e-a91f-ec3d8ad64583`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Kill (mean pos embed at the sink) vs clean, Imagewoof linear probe, 3 train-subset seeds (1000 each), fixed 1000 val; paired bootstrap over images, pooled over seeds.

RESULT (run 2e951dfb; a13f1597 killed 137 mid-pass, 42b1030b failed on a missing aimv2 registry key after finishing SigLIP2 with identical numbers):
- SigLIP2 So400m/16@256 (sink (11,0)): clean 95.2/95.4/95.5 -> kill 93.4/93.5/93.4; diff -1.93 pts [-2.87, -1.07]; per seed -1.8, -1.9, -2.1, every seed's CI excludes 0.
- AIMv2 L/14 (sink (0,0)): clean 95.0/94.9/94.8 -> kill 65.0/67.1/67.2; diff -28.47 pts [-30.87, -26.20]. (H measured -7.4 on easy Imagenette; on fine-grained Imagewoof the encoder collapses without its positional sink.)
