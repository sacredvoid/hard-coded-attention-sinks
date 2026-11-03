"""Interactive chart of per-layer outlier tokens from data/outlier-summary.json (run ac56e2a7)."""
import json
from pathlib import Path

from orx_chart import render_chart

here = Path(__file__).parent
d = json.loads((here.parent / "data" / "outlier-summary.json").read_text())["results"]
names = {"clip_l14": "CLIP L/14", "dinov2_l14": "DINOv2 L/14", "dinov2_l14_reg4": "DINOv2 L/14 +reg4",
         "siglip2_l16": "SigLIP 2 L/16", "dinov3_l16": "DINOv3 L/16", "pe_core_l14": "PE-Core L/14",
         "aimv2_l14": "AIMv2 L/14"}
count = [dict(name=names[k], points=[dict(step=i, count=c) for i, c in enumerate(v["stats"]["mean_count_gt4x"])])
         for k, v in d.items()]
prefix = [dict(name=names[k], points=[dict(step=i, prefix=max(r[i] for r in v["stats"]["prefix_ratio_mean"]))
                                      for i in range(v["num_layers"])])
          for k, v in d.items() if v["num_prefix_tokens"]]
render_chart(str(here / "outlier-tokens-by-layer.html"), title="Where high-norm tokens live, by layer",
             subtitle="1000 Imagenette val images · native resolution · one pass, no seeds · outlier = patch norm > 4× image median",
             metrics=[dict(key="count", label="Outlier patch tokens per image", unit="tokens", type="line",
                           xLabel="Block index (0 = first)", yLabel="Mean count of >4× patch tokens", series=count),
                      dict(key="prefix", label="Largest CLS/register norm", unit="× patch median", type="line",
                           yScale="log", xLabel="Block index (0 = first)",
                           yLabel="Max prefix-token norm ÷ patch median", series=prefix)],
             series=count)
