"""Claim W: find a hard-coded sink from the weights alone (no images).

Score every patch position-embedding row three ways and z-score across rows:
  norm      ||pe[i]||
  dev       ||pe[i] - mean_j pe[j]||
  neigh     ||pe[i] - mean(pe of its 4-connected grid neighbours)||  (pos embeds are smooth)
Report each score's top row, its z, and the rank/z of the empirically found sink (runs G2/K/K3).
"""

import gc
import json
import os
import shutil
import time
from pathlib import Path

import timm
import torch
from huggingface_hub import constants as hf_constants

# name -> (timm checkpoint, known sink (r, c) or None for no fixed sink)
CKPTS = {
    "aimv2_l14": ("aimv2_large_patch14_224.apple_pt", (0, 0)),
    "aimv2_l14_dist": ("aimv2_large_patch14_224.apple_pt_dist", (0, 0)),
    "aimv2_h14": ("aimv2_huge_patch14_224.apple_pt", (0, 0)),
    "aimv2_1b14": ("aimv2_1b_patch14_224.apple_pt", (0, 0)),
    "siglip1_so400m14_224": ("vit_so400m_patch14_siglip_224.webli", (12, 1)),
    "siglip1_so400m14_384": ("vit_so400m_patch14_siglip_384.webli", (21, 2)),
    "siglip2_so400m14_224": ("vit_so400m_patch14_siglip_224.v2_webli", (11, 0)),
    "siglip2_so400m16_256": ("vit_so400m_patch16_siglip_256.v2_webli", (11, 0)),
    "siglip2_so400m14_378": ("vit_so400m_patch14_siglip_378.v2_webli", (19, 0)),
    "siglip2_so400m16_384": ("vit_so400m_patch16_siglip_384.v2_webli", (16, 0)),
    "siglip2_so400m16_512": ("vit_so400m_patch16_siglip_512.v2_webli", (22, 0)),
    "siglip2_g16": ("vit_giantopt_patch16_siglip_256.v2_webli", (0, 13)),
    "paligemma1_224": ("vit_so400m_patch14_siglip_gap_224.pali_pt", (12, 1)),
    "paligemma2_3b_224": ("vit_so400m_patch14_siglip_gap_224.pali2_3b_pt", (12, 1)),
    "paligemma2_3b_448": ("vit_so400m_patch14_siglip_gap_448.pali2_3b_pt", (25, 2)),
    "siglip2_b16": ("vit_base_patch16_siglip_256.v2_webli", None),
    "siglip2_l16": ("vit_large_patch16_siglip_256.v2_webli", None),
    "siglip1_l16": ("vit_large_patch16_siglip_256.webli", None),
    "clip_l14_openai": ("vit_large_patch14_clip_224.openai", None),
}


def scores(pe, g):
    """pe: [g*g, D] patch position embeddings. Returns {name: [g*g] scores}."""
    grid = pe.view(g, g, -1)
    pad = torch.nn.functional.pad(grid.permute(2, 0, 1)[None], (1, 1, 1, 1))[0].permute(1, 2, 0)
    ones = torch.nn.functional.pad(torch.ones(g, g), (1, 1, 1, 1))
    nsum = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    ncnt = (ones[:-2, 1:-1] + ones[2:, 1:-1] + ones[1:-1, :-2] + ones[1:-1, 2:])[..., None]
    return {"norm": pe.norm(dim=-1), "dev": (pe - pe.mean(0)).norm(dim=-1),
            "neigh": (grid - nsum / ncnt).norm(dim=-1).reshape(-1)}


def z(x):
    return (x - x.mean()) / x.std()


def selftest():
    g = 5
    yy, xx = torch.meshgrid(torch.arange(g).float(), torch.arange(g).float(), indexing="ij")
    pe = torch.stack([yy, xx, yy * 0.5], -1).reshape(g * g, 3)  # smooth field
    pe[7] += torch.tensor([10.0, -10.0, 10.0])  # one hard-coded row at (1,2)
    for name, s in scores(pe, g).items():
        if name in ("dev", "neigh"):
            assert int(s.argmax()) == 7, name


def main():
    selftest()
    out, failed = {}, {}
    print("config:", json.dumps({"task": "claim_w_weight_detector", "n": len(CKPTS), "timm": timm.__version__,
                                 "commit": os.environ.get("ORX_COMMIT")}), flush=True)
    for key, (name, sink) in CKPTS.items():
        t0 = time.time()
        try:
            m = timm.create_model(name, pretrained=True, num_classes=0)
            g = m.patch_embed.grid_size[0]
            pe = m.pos_embed.detach()[0].float()
            if pe.shape[0] != g * g:  # drop prefix rows (CLIP's CLS)
                pe = pe[pe.shape[0] - g * g:]
            r = {"grid": g, "known_sink": sink}
            for sname, s in scores(pe, g).items():
                zs = z(s)
                top = int(zs.argmax())
                r[sname] = {"top": [top // g, top % g], "top_z": round(float(zs[top]), 2),
                            "second_z": round(float(zs.topk(2).values[1]), 2)}
                if sink is not None:
                    i = sink[0] * g + sink[1]
                    r[sname]["sink_z"] = round(float(zs[i]), 2)
                    r[sname]["sink_rank"] = int((zs > zs[i]).sum())
            out[key] = r
            print("RESULT_JSON " + json.dumps({"key": key, **r}) + f"  ({time.time() - t0:.0f}s)", flush=True)
            del m
            gc.collect()
            shutil.rmtree(Path(hf_constants.HF_HUB_CACHE) / f"models--timm--{name}", ignore_errors=True)
        except Exception as e:  # keep the other checkpoints
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failed[key]}", flush=True)
    print("failed:", json.dumps(failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
