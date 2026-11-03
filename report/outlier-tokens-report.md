# Claim A: outlier tokens across 7 frozen ViT-L encoders

Date: 2026-11-03. Runs: baseline `0d0d4169` (node "Outlier-token norms per layer"), positions `ac56e2a7` (child node "Outlier positions"). Both on Modal A10G, same 1000 Imagenette val images (sha1 `ed0dc412d5cb`), each model at its timm native resolution. Data: [data/outlier-summary.json](data/outlier-summary.json). Chart: [figures/outlier-tokens-by-layer.html](figures/outlier-tokens-by-layer.html).

Measure: L2 norm of every residual-stream token after each block. A patch token is an outlier if its norm exceeds 4x the median patch norm of the same image at the same layer (3x and 6x also in the JSON).

## Findings

| Encoder | Patch-token outliers? | Where the high norm goes |
|---|---|---|
| CLIP L/14 | Yes, ~2.5/image from block 11, up to ~30x, 226 distinct positions | Content-dependent patches |
| DINOv2 L/14 | Yes, ~1/image in blocks 17-21 (gone at block 23), 207 positions | Content-dependent patches |
| DINOv2 L/14 +reg4 (control) | No | Register 1 (~22x at block 12) |
| SigLIP 2 L/16 | Yes, ~2.3/image from block ~10, up to ~38x, 141 positions. Also a weak (4-6x) fixed token at (13,15) in blocks 0-4 | Content-dependent patches |
| DINOv3 L/16 | No | One register, ~2200x median at block 12, ~61x at the last block |
| PE-Core L/14 | No | CLS token, ~22x at block 12, ~6x at the last block |
| AIMv2 L/14 | Yes, but different: the top-left patch (0,0) is 100-250x in every image from block 0. Late blocks add content-dependent outliers (~16/image >4x at block 23) | Fixed position (0,0), acting like a built-in register |

## What this means for the project

- Outlier tokens are not universal in modern encoders. DINOv3 and PE-Core keep their patch tokens clean because the high-norm role sits in a register or CLS token.
- Test-time registers have a clear target only in CLIP, DINOv2, SigLIP 2 and AIMv2's late layers. AIMv2's position-(0,0) sink is a different phenomenon (fixed position, present from block 0), so the DINOv2 fix may not apply to it unchanged.
- The draft claim's "apply test-time registers unchanged to all six" should be narrowed to the four encoders with patch outliers, with DINOv3 and PE-Core as encoders that are already clean.

## Caveats

- One pass over 1000 images from 10 Imagenette classes, no repeated sampling. The 0.0039 = 1/256 fractions are exact counts, not noise, but other image domains are untested.
- Resolutions differ (224 to 518) because each model runs at its native size. DINOv2 at 518 has 1369 patches, so its per-token fractions are not directly comparable. Use per-image counts.
- Relative threshold (4x the image median). Counts barely change between 3x and 6x for CLIP, DINOv2 and SigLIP 2. AIMv2's late-layer count does change (17 at 3x vs 7 at 6x), so its late spread is partly a broad norm increase.
- timm ports of each model. Hooks read block outputs before the final norm.
