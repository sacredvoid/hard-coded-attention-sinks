# 23-D-register-takeover: D: can one constant extra token take over the hard-coded sink at test time?

Branch snapshot of experiment `5907dc4f-08b3-4348-bbf7-2ae2ac5ac827`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Kill the pos-embed sink and append ONE constant token before block 0 = mean clean block-0 input at the sink patch over 200 calibration images (disjoint from probe train). No training. Imagewoof probe 1000/1000; features exclude sink patch and register; paired bootstrap vs clean.

RESULT (run 2db55c09): a constant extra token takes over the sink.
- AIMv2 L: clean 95.0; kill 65.0 (-30.0); kill+reg 94.9 (-0.1 [-0.6,+0.4]) = full recovery; clean+reg 95.0 (+0.0); kill+zero 93.3 (-1.7 [-2.7,-0.7]). Norm ratio at mid block: register 224.7x (= the clean sink's 228.5x), sink patch falls to 72.8x.
- SigLIP2 So400m: clean 95.2; kill 93.4 (-1.8 [-2.9,-0.7]); kill+reg 94.6 (-0.6 [-1.5,+0.3]); clean+reg 95.2 (+0.0); kill+zero 93.3 (-1.9, no help). Register becomes the sink (146.9x mid); patch (11,0) returns to normal (2.5x).
=> the hard-coded sink can be moved off the image grid at test time with no retraining; AIMv2 even recovers most of the way with a zero token, SigLIP2 needs the sink's actual value.
