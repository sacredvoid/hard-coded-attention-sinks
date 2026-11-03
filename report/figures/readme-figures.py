"""Static README figures. Regenerate: python readme-figures.py

Data: ../data/readme-figures.json (swap: runs G2 8048b620 + K2 c53559d4; cliff: A1 bfdf36d2 + A2
2e1807d8; VLM deltas: V2 7eb05244, V3 5365c0a1, VV a66c6957) and ../data/sink-attention.json
(runs G 9aa6e442 + K2 c53559d4).
"""

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from orx_figstyle import BASELINE, COLUMN, PALETTE, SEQUENTIAL, TEXT, figure, figure_grid, save, use_style

HERE = Path(__file__).parent
D = json.loads((HERE.parent / "data" / "readme-figures.json").read_text())
ATT = json.loads((HERE.parent / "data" / "sink-attention.json").read_text())["models"]


def sink_swap():
    rows = [("aimv2_l14", "AIMv2 L/14", 12), ("siglip2_so400m16", "SigLIP 2 So400m/16", 20)]
    fig, axes = figure_grid(2, 2, width=TEXT, ratio=1.0)
    cmap = mpl.colormaps[SEQUENTIAL]
    for r, (key, name, layer) in enumerate(rows):
        e = D["pos_swap"][key]
        g = e["grid"]
        for c, variant in enumerate(("clean", "swap")):
            v = e[variant]
            share_sink, share_tgt = v["top_at_sink"][layer], v["top_at_target"][layer]
            assert share_sink + share_tgt > 0.99, (key, variant, share_sink, share_tgt)  # other cells are then 0
            grid = np.zeros((g, g))
            grid[tuple(e["sink"])] = share_sink
            grid[tuple(e["target"])] = share_tgt
            ax = axes[r, c]
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            for (rr, cc), lab in ((e["sink"], "sink pos. emb."), (e["target"], "(8,8)")):
                ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=0.8))
            hot = e["sink"] if variant == "clean" else e["target"]
            mass = v["attn_sink"][layer] if variant == "clean" else v["attn_target"][layer]
            ax.annotate(f"{100 * max(share_sink, share_tgt):.0f}% of images\nattention {mass:.2f}",
                        xy=(hot[1], hot[0]), xytext=(min(hot[1] + 2.0, 9.5), hot[0] + 3.5 if hot[0] < 8 else hot[0] - 3.0),
                        color="white", fontsize=6.5, arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
            ax.set_xticks([0, 8, 15])
            ax.set_yticks([0, 8, 15])
            ax.tick_params(length=0, labelsize=6)
            ax.grid(False)
            ax.set_xlabel("patch column", fontsize=6.5)
            ax.set_ylabel("patch row", fontsize=6.5)
            cond = "original" if variant == "clean" else "pos. embedding swapped with (8,8)"
            ax.text(0.5, 1.04, f"{name}, block {layer}: {cond}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6.5)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.03)
    cb.set_label("share of images whose top-norm patch is here", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, str(HERE / "sink-swap"))


GROUPS = {  # sink location, from runs G / G2 / K / K2 / K5
    "content patches": (PALETTE["orange"], ["clip_l14", "dinov2_l14", "siglip2_l16", "clip_l14_laion", "pe_core_g14"]),
    "one fixed patch": (PALETTE["blue"], ["aimv2_l14", "siglip2_so400m16", "siglip2_g16"]),
    "register / CLS": (PALETTE["green"], ["dinov2_l14_reg4", "dinov3_l16", "pe_core_l14"]),
    "none found": (BASELINE, ["clip_h14_laion"]),
}


def sink_by_depth():
    fig, ax = figure(width=TEXT, ratio=0.5)
    for group, (color, keys) in GROUPS.items():
        for k in keys:
            y = np.array(ATT[k]["top_key_mass"])
            x = 100 * np.arange(len(y)) / (len(y) - 1)
            ax.plot(x, y, color=color, linewidth=1.0, alpha=0.85)
    ax.set_xlabel("relative depth (% of blocks)")
    ax.set_ylabel("attention on the most-attended key")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.72)
    handles = [Line2D([], [], color=c, linewidth=1.5, label=f"{g} ({len(k)})") for g, (c, k) in GROUPS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=6.5, frameon=True, framealpha=1.0, edgecolor="black")
    save(fig, str(HERE / "sink-by-depth"))


def aimv2_cliff():
    pts = sorted((float(a), v) for a, v in D["aimv2_cliff"].items())
    clean = pts[0][1]["acc"] * 100
    fig, ax = figure(width=COLUMN, ratio=0.68)
    ax.axhline(clean, color=BASELINE, linewidth=0.6, linestyle=":")
    x = [a for a, _ in pts]
    y = [v["acc"] * 100 for _, v in pts]
    lo = [clean + v["vs_clean"][1] * 100 for _, v in pts]
    hi = [clean + v["vs_clean"][2] * 100 for _, v in pts]
    ax.fill_between(x, lo, hi, color=PALETTE["red"], alpha=0.18, linewidth=0)
    ax.plot(x, y, "o-", color=PALETTE["red"], markersize=3, linewidth=1.0)
    ax.set_xlabel("sink pos. embedding moved toward mean")
    ax.set_ylabel("Imagewoof probe accuracy (%)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 100)
    ax.text(0.02, clean + 1.5, "clean", fontsize=6.5, color=BASELINE)
    save(fig, str(HERE / "aimv2-cliff"))


def vlm_effects():
    rows = []
    for setting, d in D["vlm_deltas"].items():
        for var, label in (("full_minus_sink", "drop the sink's output token"), ("kill_full", "kill the sink inside the encoder")):
            rows.append((f"{setting}: {label}", *d[var]["diff"], var))
    fig, ax = figure(width=TEXT, ratio=0.36)
    ax.axvline(0, color=BASELINE, linewidth=0.8, zorder=1)
    for i, (label, m, lo, hi, var) in enumerate(rows[::-1]):
        color = PALETTE["red"] if var == "kill_full" else PALETTE["blue"]
        ax.plot([lo * 100, hi * 100], [i, i], color=color, linewidth=1.2, solid_capstyle="round")
        ax.plot([m * 100], [i], "o", color=color, markersize=3.5)
        ax.annotate(f"{m * 100:+.1f}", xy=(hi * 100, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=color)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=6.5)
    ax.set_xlabel("accuracy change vs all tokens (points), 95% CI")
    ax.set_xlim(-3.2, 2.0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, str(HERE / "vlm-effects"))


if __name__ == "__main__":
    use_style()
    sink_swap()
    sink_by_depth()
    aimv2_cliff()
    vlm_effects()
