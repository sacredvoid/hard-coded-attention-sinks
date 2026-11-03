# 22-A3-siglip2-key-value: A3: does SigLIP2 So400m's sink also work only through its key/value?

Branch snapshot of experiment `b4ce3ff1-c7e9-4877-a4f4-e8b070fbc98c`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

A2's key/value test on SigLIP2 So400m/16@256, sink = patch (11,0) (index 176). Imagewoof probe 1000/1000, features = mean of all patches except the sink; paired bootstrap vs clean.

RESULT (run 626e2a34): same mechanism as AIMv2. clean 95.2; kill 93.4 (-1.8 [-2.9,-0.7], matches K4-CI -1.9); kv@0 (clean key+value of the sink in block 0 only, its hidden state still killed) 95.1 (-0.1 [-1.0,+0.8]) = full rescue; kv@all 95.2 (+0.0, identical predictions); q@0 control 92.8 (-2.4 [-3.7,-1.2]), no rescue. The sink's function for the rest of the image runs through its key/value, already from block 0.
