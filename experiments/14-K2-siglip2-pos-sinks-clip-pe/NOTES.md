# 14-K2-siglip2-pos-sinks-clip-pe: K2: are big-SigLIP2 fixed sinks position-embedding sinks? + where CLIP-H/PE-G sink

Branch snapshot of experiment `e3a6ba40-69fd-408c-b770-e9437ef61aaf`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

SigLIP2 So400m/g fixed sinks: swap their pos embed with (8,8); attention anatomy for CLIP-H LAION, CLIP-L LAION, PE-Core G. 200 val imgs, A10G.

RESULT (run c53559d4; first attempt 5ca60ccd killed while downloading PE-G, its first 3 models matched): big-SigLIP2 fixed sinks ARE position-embedding sinks: swapping the (11,0) [So400m] or (0,13) [g] pos embed with (8,8) moves the top-norm token AND attention mass to (8,8) in 100% of images at every layer (So400m L20 attn 0.447 -> 0.445 at new spot; g L26 0.500 -> 0.498). Together with G2 (AIMv2): positional sinks are a general mechanism, not specific to autoregressive training.
PE-Core G (no CLS): ~12.6 patch outliers/img collectively take 44-67% of attention from L35 (distributed sink, each token ~5%).
OpenCLIP L/14 LAION: weaker content outliers (3.3/img, 17-20% attention). OpenCLIP H/14 LAION: NO detectable sink (top key mass 2.5-6% after L12, entropy 4.5-5.3) -> the one sinkless encoder found so far.
