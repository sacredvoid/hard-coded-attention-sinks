"""Claim K2: are the fixed sinks in big SigLIP2 models written by one position embedding?

SigLIP2 So400m (fixed sink at (11,0)) and g (fixed sink at (0,13)), found in run K: swap that
position embedding with the one at (8,8) and track, per layer, the share of images whose
top-norm patch is at the original / target position and the attention mass on each.
Attention anatomy (attention_sinks.Probe) for those two plus CLIP H/14 LAION (no high-norm
token in run K) and PE-Core G (patch outliers, no CLS).
"""

import gc
import shutil
from pathlib import Path
import hashlib
import json
import os
import time

import timm
import torch
from PIL import Image

from attention_sinks import Probe, show
from huggingface_hub import constants as hf_constants

from measure_outliers import MODELS, image_paths

# CLIP-H LAION showed no sink in run 5ca60ccd; check the LAION ViT-L too
MODELS["clip_l14_laion"] = "vit_large_patch14_clip_224.laion2b"

N_IMG, SEED = 200, 0
FIXED = {"siglip2_so400m16": (11, 0), "siglip2_g16": (0, 13)}  # from run K
ANATOMY = ["siglip2_so400m16", "siglip2_g16", "clip_h14_laion", "clip_l14_laion", "pe_core_g14"]


@torch.no_grad()
def track(pr, paths, a, b):
    """Per layer: share of images with top-norm patch at a / b, and attention mass on a / b."""
    L, npt = pr.L, pr.npt
    acc = {k: [0.0] * L for k in ("top_a", "top_b", "attn_a", "attn_b")}
    hooks = []

    def blk(l):
        def fn(m, i, o):
            top = o[:, npt:].float().norm(dim=-1).argmax(1)
            acc["top_a"][l] += (top == a).sum().item()
            acc["top_b"][l] += (top == b).sum().item()
        return fn

    def att(l):
        def fn(m, i, w):
            k = w[:, :, npt:].float().mean(1).mean(1)
            acc["attn_a"][l] += k[:, npt + a].sum().item()
            acc["attn_b"][l] += k[:, npt + b].sum().item()
        return fn

    for l, blkm in enumerate(pr.model.blocks):
        hooks += [blkm.register_forward_hook(blk(l)), blkm.attn.attn_drop.register_forward_hook(att(l))]
    n = 0
    for s in range(0, len(paths), pr.bs):
        x = torch.stack([pr.tf(Image.open(p).convert("RGB")) for p in paths[s:s + pr.bs]]).to(pr.device)
        pr.model.forward_features(x)
        n += len(x)
    for h in hooks:
        h.remove()
    return {k: [v / n for v in arr] for k, arr in acc.items()}


def free(pr, key):
    """Break hook ref cycles and drop the checkpoint from disk (first attempt was killed mid-download)."""
    shutil.rmtree(Path(hf_constants.HF_HUB_CACHE) / f"models--timm--{MODELS[key]}", ignore_errors=True)
    for m in pr.model.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()
    del pr.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = image_paths(N_IMG, SEED)  # val split
    config = {"task": "claim_k2_positional_sinks", "n_img": N_IMG, "fixed": FIXED, "anatomy": ANATOMY,
              "device": device, "torch": torch.__version__, "timm": timm.__version__,
              "commit": os.environ.get("ORX_COMMIT"),
              "images_sha1": hashlib.sha1("\n".join(p.name for p in paths).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    failed = {}
    for key in ANATOMY:
        t0 = time.time()
        try:
            pr = Probe(key, device)
            print(f"\n== {key} grid={pr.g} prefix={pr.npt} layers={pr.L}", flush=True)
            r = pr.run(paths)
            show(key, "clean", r, pr.L)
            print("RESULT_JSON " + json.dumps({"key": key, "variant": "anatomy", **r}), flush=True)
            if key in FIXED:
                g = pr.g
                a = FIXED[key][0] * g + FIXED[key][1]
                b = 8 * g + 8
                pe = pr.model.pos_embed.data
                orig = pe.clone()
                for name in ("clean", "swap"):
                    pe.copy_(orig)
                    if name == "swap":
                        pe[0, a], pe[0, b] = orig[0, b], orig[0, a]
                    t = track(pr, paths, a, b)
                    print(f"  [{name}] pos {FIXED[key]} vs (8,8) layer: top@orig top@(8,8) | attn_orig attn_(8,8)",
                          flush=True)
                    for l in range(pr.L):
                        print(f"  {l:2d}: {t['top_a'][l]:.2f} {t['top_b'][l]:.2f} | "
                              f"{t['attn_a'][l]:.3f} {t['attn_b'][l]:.3f}", flush=True)
                    print("RESULT_JSON " + json.dumps({"key": key, "variant": f"pos_{name}", **t}), flush=True)
                pe.copy_(orig)
            free(pr, key)
            print(f"{key} done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep other encoders' results
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failed[key]}", flush=True)
    print("failed:", json.dumps(failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
