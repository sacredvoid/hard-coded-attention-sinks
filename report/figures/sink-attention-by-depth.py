"""Attention-sink strength by relative depth for 12 frozen encoders.

Data: ../data/sink-attention.json (extracted from runs G 9aa6e442 and K2 c53559d4).
Run: python sink-attention-by-depth.py  (writes sink-attention-by-depth.html beside it)
"""

import json
from pathlib import Path

from orx_chart import render_chart

HERE = Path(__file__).parent
data = json.loads((HERE.parent / "data" / "sink-attention.json").read_text())["models"]

LABELS = {  # name, sink location (from runs G, G2, K, K2)
    "clip_l14": "CLIP L/14 (OpenAI) · content patches",
    "dinov2_l14": "DINOv2 L/14 · content patches",
    "siglip2_l16": "SigLIP 2 L/16 · content patches",
    "clip_l14_laion": "OpenCLIP L/14 (LAION) · content patches",
    "pe_core_g14": "PE-Core G/14 (no CLS) · content patches",
    "aimv2_l14": "AIMv2 L/14 · fixed patch (0,0)",
    "siglip2_so400m16": "SigLIP 2 So400m/16 · fixed patch (11,0)",
    "siglip2_g16": "SigLIP 2 g/16 · fixed patch (0,13)",
    "dinov2_l14_reg4": "DINOv2 L/14 + 4 reg · register",
    "dinov3_l16": "DINOv3 L/16 · register",
    "pe_core_l14": "PE-Core L/14 · CLS",
    "clip_h14_laion": "OpenCLIP H/14 (LAION) · none found",
}
DASH = {"content": None, "fixed": "6 3", "register": "2 3", "CLS": "2 3", "none": "1 4"}


def series(metric):
    out = []
    for key, label in LABELS.items():
        vals = data[key][metric]
        n = len(vals) - 1
        kind = label.split("· ")[1].split()[0]
        pts = [{"step": round(100 * l / n, 1), metric: v} for l, v in enumerate(vals)]
        out.append({"name": label, "points": pts, **({"dash": DASH[kind]} if DASH.get(kind) else {})})
    return out


render_chart(
    str(HERE / "sink-attention-by-depth.html"),
    title="Where the attention sink lives, by depth",
    subtitle="Patch-query attention, head-mean · 300 (G) or 200 (K2) Imagenette val images · single pass, no seeds",
    metrics=[
        dict(key="top_key_mass", label="Mass on the single most-attended key", type="line", unit="fraction",
             xLabel="relative depth (% of blocks)", yLabel="attention mass", series=series("top_key_mass")),
        dict(key="outlier_mass", label="Mass on all high-norm patch tokens (>4x median)", type="line",
             unit="fraction", xLabel="relative depth (% of blocks)", yLabel="attention mass",
             series=series("outlier_mass")),
    ],
    x_label="relative depth (% of blocks)",
)
print("wrote", HERE / "sink-attention-by-depth.html")
