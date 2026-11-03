"""Claim G2: is AIMv2's (0,0) sink set by its position embedding? Massive channels per encoder.

Part 1 (AIMv2): edit pos_embed and see where the sink goes. Variants: clean; swap the (0,0)
position embedding with the one at (8,8) or (15,15); replace (0,0)'s with the mean position
embedding. Per layer: share of images whose top-norm patch is at (0,0) / at the swap target,
and attention mass (patch queries, head-mean) on each.
Part 2 (all 7): at each encoder's sink token (from run G), mean |activation| per channel.
Top channels vs the median channel of normal patches, and how often each image's top-1
channel matches the encoder-wide top-1 (fixed 'massive' dims, as in LLMs).
"""

import gc
import hashlib
import json
import os
import time

import timm
import torch
from PIL import Image

from attention_sinks import Probe
from measure_outliers import MODELS, image_paths

N_IMG, SEED = 200, 0


class _Stop(Exception):
    pass

# sink token per encoder from run G: ("patch", layer) = top-norm patch; ("prefix", layer, idx)
SINK = {"clip_l14": ("patch", 16), "dinov2_l14": ("patch", 20), "dinov2_l14_reg4": ("prefix", 20, 2),
        "siglip2_l16": ("patch", 16), "dinov3_l16": ("prefix", 20, 3), "pe_core_l14": ("prefix", 19, 0),
        "aimv2_l14": ("patch", 4)}


class Plain:
    """Fused-attention model with the fields massive_channels needs (no attention hooks)."""

    def __init__(self, key, device):
        self.device = device
        self.model = timm.create_model(MODELS[key], pretrained=True, num_classes=0).eval().to(device)
        self.tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=self.model))
        self.npt = int(getattr(self.model, "num_prefix_tokens", 0))
        self.bs = 8


def free(obj):
    """Probe hooks form reference cycles (module -> hook -> Probe -> model); break them."""
    for m in obj.model.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()
    del obj.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batches(pr, paths):
    for s in range(0, len(paths), pr.bs):
        yield torch.stack([pr.tf(Image.open(p).convert("RGB")) for p in paths[s:s + pr.bs]]).to(pr.device)


@torch.no_grad()
def where_is_sink(pr, paths, target):
    L, npt = pr.L, pr.npt
    top00, toptg, m00, mtg = [0.0] * L, [0.0] * L, [0.0] * L, [0.0] * L
    hooks = []

    def blk(l):
        def fn(m, i, o):
            top = o[:, npt:].float().norm(dim=-1).argmax(1)
            top00[l] += (top == 0).sum().item()
            toptg[l] += (top == target).sum().item()
        return fn

    def att(l):
        def fn(m, i, a):
            k = a[:, :, npt:].float().mean(1).mean(1)  # [b, T]
            m00[l] += k[:, npt].sum().item()
            mtg[l] += k[:, npt + target].sum().item()
        return fn

    for l, b in enumerate(pr.model.blocks):
        hooks += [b.register_forward_hook(blk(l)), b.attn.attn_drop.register_forward_hook(att(l))]
    n = 0
    for x in batches(pr, paths):
        pr.model.forward_features(x)
        n += len(x)
    for h in hooks:
        h.remove()
    return {k: [v / n for v in arr] for k, arr in
            (("top_at_00", top00), ("top_at_target", toptg), ("attn_00", m00), ("attn_target", mtg))}


@torch.no_grad()
def massive_channels(pr, key, paths):
    spec = SINK[key]
    l, npt = spec[1], pr.npt
    acts, normal = [], []

    def fn(m, i, o):
        o = o.float()
        if spec[0] == "patch":
            idx = o[:, npt:].norm(dim=-1).argmax(1) + npt
        else:
            idx = torch.full((len(o),), spec[2], device=o.device)
        ar = torch.arange(len(o), device=o.device)
        acts.append(o[ar, idx].abs().cpu())
        normal.append(o[:, npt:].abs().median(1).values.cpu())  # per-channel median over patches
        raise _Stop  # nothing after the sink layer is needed

    h = pr.model.blocks[l].register_forward_hook(fn)
    for x in batches(pr, paths):
        try:
            pr.model.forward_features(x)
        except _Stop:
            pass
    h.remove()
    a, nm = torch.cat(acts), torch.cat(normal)  # [n, D]
    mean_a = a.mean(0)
    base = nm.median(1).values.mean()  # typical channel magnitude of a patch
    top = mean_a.argsort(descending=True)[:5]
    top1_img = a.argmax(1)
    return {"layer": l, "top_channels": top.tolist(), "top_ratio_vs_patch_median": (mean_a[top] / base).tolist(),
            "share_imgs_top1_is_global_top1": (top1_img == top[0]).float().mean().item(),
            "share_imgs_top1_in_global_top5": torch.isin(top1_img, top).float().mean().item(),
            "sink_norm_share_top5": (a[:, top].pow(2).sum(1) / a.pow(2).sum(1)).mean().item()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = image_paths(N_IMG, SEED)  # val split
    config = {"task": "claim_g2_sink_mechanism", "n_img": N_IMG, "sink": SINK, "device": device,
              "torch": torch.__version__, "timm": timm.__version__, "commit": os.environ.get("ORX_COMMIT"),
              "images_sha1": hashlib.sha1("\n".join(p.name for p in paths).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    results, failed = {"massive": {}}, {}
    try:
        pr = Probe("aimv2_l14", device)
        g = pr.g
        pe = pr.model.pos_embed.data
        orig = pe.clone()
        variants = {"clean": None, "swap_8_8": 8 * g + 8, "swap_15_15": 15 * g + 15, "mean_pe_at_00": "mean"}
        results["aimv2_pos"] = {}
        for name, v in variants.items():
            pe.copy_(orig)
            target = 8 * g + 8 if v in (None, "mean") else v
            if isinstance(v, int):
                pe[0, 0], pe[0, v] = orig[0, v], orig[0, 0]
            elif v == "mean":
                pe[0, 0] = orig[0].mean(0)
            r = where_is_sink(pr, paths, target)
            results["aimv2_pos"][name] = {"target": target, **r}
            print(f"\n[aimv2 {name}] target={target} layer: top@00 top@target | attn00 attn_target", flush=True)
            for l in range(pr.L):
                print(f"  {l:2d}: {r['top_at_00'][l]:.2f} {r['top_at_target'][l]:.2f} | "
                      f"{r['attn_00'][l]:.3f} {r['attn_target'][l]:.3f}", flush=True)
        pe.copy_(orig)
        free(pr)
    except Exception as e:
        failed["aimv2_pos"] = f"{type(e).__name__}: {e}"
        print(f"!! aimv2_pos failed: {failed['aimv2_pos']}", flush=True)
    for key in MODELS:
        t0 = time.time()
        try:
            pr = Plain(key, device)
            r = massive_channels(pr, key, paths)
            results["massive"][key] = r
            print(f"\n[massive {key}] L{r['layer']} top={r['top_channels']} "
                  f"ratio={[round(v, 1) for v in r['top_ratio_vs_patch_median']]} "
                  f"top1=global:{r['share_imgs_top1_is_global_top1']:.2f} in_top5:{r['share_imgs_top1_in_global_top5']:.2f} "
                  f"energy_top5={r['sink_norm_share_top5']:.2f} ({time.time() - t0:.0f}s)", flush=True)
            free(pr)
        except Exception as e:
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failed[key]}", flush=True)
    print("\n=== SUMMARY_JSON_BEGIN ===")
    print(json.dumps({"config": config, "results": results, "failed": failed}))
    print("=== SUMMARY_JSON_END ===")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
