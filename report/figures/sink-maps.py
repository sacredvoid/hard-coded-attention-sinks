"""Per-image sink maps on real photos. Regenerate: python sink-maps.py

Data: ../data/sink-maps.json + photos/img*.jpg, extracted from run FIG (3654a9c2). Each map is the
per-patch norm / median patch norm at one block; all models saw the same square crop.
Photo 2 (shows an identifiable person) is deliberately left out.
"""

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Rectangle
from orx_figstyle import SEQUENTIAL, WIDE, figure_grid, save, use_style
from PIL import Image

HERE = Path(__file__).parent
IMGS = [0, 1, 3]
COLS = [  # (model, variant, column title)
    ("clip_l14", "clean", "OpenAI CLIP L\nblock 16"),
    ("dinov2_l14", "clean", "DINOv2 L\nblock 20"),
    ("dinov3_l16", "clean", "DINOv3 L\nblock 20"),
    ("aimv2_l14", "clean", "AIMv2 L\nblock 12"),
    ("aimv2_l14", "swap", "AIMv2 L, pos. emb.\nswapped to (8,8)"),
    ("siglip2_so400m16", "clean", "SigLIP 2 So400m\nblock 20"),
    ("siglip2_so400m16", "swap", "SigLIP 2 So400m, pos.\nemb. swapped to (8,8)"),
]
VMAX = 250.0


def main():
    use_style()
    data = json.loads((HERE.parent / "data" / "sink-maps.json").read_text())["maps"]
    by = {(d["img"], d["model"], d["variant"]): d for d in data}
    norm = mpl.colors.LogNorm(vmin=1.0, vmax=VMAX)
    cmap = mpl.colormaps[SEQUENTIAL]
    fig, axes = figure_grid(len(IMGS), len(COLS) + 1, width=WIDE, ratio=len(IMGS) / (len(COLS) + 1) * 1.18)
    for r, i in enumerate(IMGS):
        photo = np.asarray(Image.open(HERE / "photos" / f"img{i}.jpg"))
        axes[r, 0].imshow(photo)
        for c, (model, variant, title) in enumerate(COLS, start=1):
            d = by[(i, model, variant)]
            m = np.clip(np.array(d["map"]), 1.0, VMAX)
            g = d["grid"]
            ax = axes[r, c]
            ax.imshow(photo, extent=(0, g, g, 0))
            shown = np.ma.masked_less(m, 2.0)  # colour only patches above 2x the median; photo stays visible
            ax.imshow(shown, cmap=cmap, norm=norm, alpha=0.9, extent=(0, g, g, 0), interpolation="nearest")
            tr, tc = d["top"]
            if d["top_ratio"] >= 4:  # only box a real outlier
                ax.add_patch(Rectangle((tc, tr), 1, 1, fill=False, edgecolor="white", linewidth=0.9))
            ax.text(0.97, 0.03, f"max {d['top_ratio']:.0f}x" if d["top_ratio"] >= 4 else f"max {d['top_ratio']:.1f}x",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, linewidth=0))
            if r == 0:
                ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5)
        if r == 0:
            axes[r, 0].text(0.5, 1.04, "photo\n(same crop for all)", transform=axes[r, 0].transAxes,
                            ha="center", va="bottom", fontsize=5.5)
    for ax in axes.flat:
        ax.axis("off")  # photo grid: no data axes to label
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("patch norm / median patch norm (log; < 2x not coloured)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    save(fig, str(HERE / "sink-maps"), formats=("pdf",), close=False)
    fig.savefig(HERE / "sink-maps.jpg", dpi=200, pil_kwargs={"quality": 88})  # photos: compact raster for the README


if __name__ == "__main__":
    main()
