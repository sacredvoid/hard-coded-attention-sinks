# 18-K5-open-data-clip: K5: is the sinkless OpenCLIP-H a LAION quirk or a scale trend?

Branch snapshot of experiment `3496bc3f-a86b-4da7-b129-b1661a54172b`. Run with `uv run --locked python measure_outliers.py` (needs a GPU; we used Modal T4/A10G).

## Notes and result

Attention anatomy (200 Imagenette val imgs, A10G) for CLIP-style L/H/G trained on LAION / DFN / MetaCLIP.

RESULT (run 491db036): weak or absent sinks are common across open-data CLIPs, not a LAION-H quirk. Late-half peak top-key mass: L/14 LAION-400M 0.09, DFN-2B 0.05 (3.7 outliers/img taking 13%), MetaCLIP-2.5B 0.18 (outliers 28%); H/14 DFN-5B 0.05 (0 outliers), MetaCLIP 0.09 (0.1 outliers); G/14 LAION-2B 0.13 (on CLS, 2.5 outliers/img holding only 3.5%), MetaCLIP 0.045. Compare OpenAI CLIP L/14 0.28-0.31 late-block (G, G-rep) and LAION-2B L/14 outliers 20% (K2). Sink strength falls with scale within MetaCLIP (0.18 -> 0.09 -> 0.045) and DFN; late attention entropy stays high (3.7-4.4 nats).
