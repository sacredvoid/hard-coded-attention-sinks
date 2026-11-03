"""The hard-coded sink works only through its key/value. Regenerate: python sink-mechanism.py

Data: ../data/sink-mechanism.csv, extracted from runs A2 (2e1807d8), A3 (626e2a34), D (2db55c09).
Imagewoof linear probe, 1000 train / 1000 val images, single probe seed per condition.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from orx_figstyle import BASELINE, MUTED, PALETTE, TEXT, figure, save, use_style

HERE = Path(__file__).parent
MODELS = ["AIMv2 L/14", "SigLIP 2 So400m/16"]
# fixed order; grey = reference, red = broken, blue = repairs, light = control
CONDS = [
    ("clean", MUTED),
    ("sink killed", PALETTE["red"]),
    ("query restored (control)", "#E8B4B4"),
    ("key/value restored, block 0", "#9DBBE0"),
    ("key/value restored, all blocks", PALETTE["blue"]),
    ("killed + constant token", PALETTE["green"]),
]


def main():
    use_style()
    acc = defaultdict(dict)
    with open(HERE.parent / "data" / "sink-mechanism.csv") as f:
        for r in csv.DictReader(f):
            acc[r["model"]][r["condition"]] = float(r["accuracy"])
    fig, ax = figure(width=TEXT, ratio=0.42)
    x = np.arange(len(MODELS))
    w = 0.9 / len(CONDS)
    for i, (cond, color) in enumerate(CONDS):
        off = (i - (len(CONDS) - 1) / 2) * w
        bars = ax.bar(x + off, [acc[m][cond] for m in MODELS], w, label=cond, color=color,
                      edgecolor="black", linewidth=0.4)
        ax.bar_label(bars, fmt="%.1f", fontsize=5, padding=1.5)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("Imagewoof probe accuracy (%)")
    top = 10 * math.ceil(max(max(v.values()) for v in acc.values()) * 1.12 / 10)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, top + 1, 20))
    ax.tick_params(axis="x", length=0)
    ax.axhline(acc[MODELS[0]]["clean"], color=BASELINE, linewidth=0.5, linestyle=":", zorder=0)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6.5, frameon=False)
    save(fig, str(HERE / "sink-mechanism"))


if __name__ == "__main__":
    main()
