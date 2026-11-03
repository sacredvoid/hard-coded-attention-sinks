# 29-FIG: per-image sink maps on real photos

Branch snapshot of the figure run. Run with `uv run --locked python measure_outliers.py` (GPU). Renders with `report/figures/sink-maps.py`.

## Notes and result

Visual evidence for the README: 4 Imagenette val photos (seed 3), one shared square crop resized to each model's input; per-patch norm / median patch norm at one block. Photos (base64 JPEG) and maps printed to the log.

RESULT (run 3654a9c2): AIMv2 L block 12: top patch (0,0) in 4/4 photos (220-238x); with the (0,0) pos embed swapped to (8,8): top patch (8,8) in 4/4 (219-239x). SigLIP2 So400m block 20: (11,0) in 4/4 (70-78x); swapped: (8,8) in 4/4 (68-75x). OpenAI CLIP L block 16: top patch differs per photo (24-26x). DINOv2 L block 20: differs per photo (19-21x). DINOv3 L block 20: no patch outlier (max 1.2-1.6x; its sink is a register). Figure: report/figures/sink-maps.jpg (photo 2 excluded from the figure: identifiable person).
