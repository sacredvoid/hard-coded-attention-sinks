# 21-A2-aimv2-key-value: A2: does AIMv2 need the sink's key/value in block 0? + where is the cliff

Branch snapshot of experiment `7d06b876-1fd7-4277-9e17-6d998c4032cc`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

AIMv2 L/14 killed (mean pos embed at (0,0)); Imagewoof probe as A1; paired bootstrap vs clean.

RESULT (run 2e1807d8): the sink works entirely through its key/value.
- Cliff edge (pos embed blended toward the mean): 0.75 -0.1 (A1), 0.8 -2.3 [-3.5,-1.3], 0.9 -8.2 [-10.1,-6.4], 0.95 -26.9 [-29.8,-24.0], 1.0 -30.0.
- kv0@0 (clean key+value of token 0 in block 0 only, token 0's hidden state still killed): 90.8, -4.2 [-5.6,-2.8] -> recovers 85% of the 30-pt gap.
- kv0@all (clean key+value of token 0 at every block): 95.0, +0.0 [+0.0,+0.0] -> full rescue, predictions identical to clean.
- q0@0 control (only token 0's query): 65.5, no rescue.
The other patches need to read the sink's K/V; the sink token's own state is irrelevant to the image representation.
