"""Claim K4: remove the hard-coded sink in SigLIP So400m towers. What breaks, what moves?

Sinks from run K3: SigLIP v1 So400m/14@384 at (21,2); SigLIP2 So400m/16@256 at (11,0).
Variants: clean; swap (sink pos embed <-> grid-centre pos embed); kill (mean pos embed at
the sink patch). Per layer on val images: outliers (norm > K x median) per image, share of
images whose top-norm patch is the sink patch, attention mass (patch queries, head-mean) on
the sink patch and on the most-attended key. Imagewoof linear probe on mean-pooled final
patches per variant.
"""

import gc
import hashlib
import json
import os
import random
import time

import timm
import torch
from PIL import Image

from attention_sinks import Probe
from measure_outliers import MODELS, PRIMARY_K, image_paths
from probe import linear_probe

MODELS["siglip1_so400m14_384"] = "vit_so400m_patch14_siglip_384.webli"
MODELS["siglip2_so400m16_256"] = "vit_so400m_patch16_siglip_256.v2_webli"
SINK = {"siglip1_so400m14_384": (21, 2), "siglip2_so400m16_256": (11, 0)}  # from runs K3 / K
DATASET, N_TR, N_TE, SEED = "imagewoof2-320", 1000, 1000, 0


@torch.no_grad()
def run(pr, paths, sink, stats):
    L, npt = pr.L, pr.npt
    acc = {k: [0.0] * L for k in ("outliers", "top_is_sink", "attn_sink", "attn_top")}
    hooks = []
    if stats:
        def blk(l):
            def fn(m, i, o):
                n = o[:, npt:].float().norm(dim=-1)
                r = n / n.median(dim=1, keepdim=True).values
                acc["outliers"][l] += (r > PRIMARY_K).sum().item()
                acc["top_is_sink"][l] += (r.argmax(1) == sink).sum().item()
            return fn

        def att(l):
            def fn(m, i, w):
                k = w[:, :, npt:].float().mean(1).mean(1)
                acc["attn_sink"][l] += k[:, npt + sink].sum().item()
                acc["attn_top"][l] += k.max(1).values.sum().item()
            return fn

        for l, b in enumerate(pr.model.blocks):
            hooks += [b.register_forward_hook(blk(l)), b.attn.attn_drop.register_forward_hook(att(l))]
    feats = []
    for s in range(0, len(paths), pr.bs):
        x = torch.stack([pr.tf(Image.open(p).convert("RGB")) for p in paths[s:s + pr.bs]]).to(pr.device)
        feats.append(pr.model.forward_features(x)[:, npt:].float().mean(1).cpu())
    for h in hooks:
        h.remove()
    n = len(paths)
    return torch.cat(feats), ({k: [v / n for v in arr] for k, arr in acc.items()} if stats else None)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = image_paths(N_TR, SEED, "train", DATASET)
    random.Random(SEED).shuffle(tr)
    te = image_paths(N_TE, SEED, "val", DATASET)
    classes = sorted({p.parent.name for p in tr})
    if {p.parent.name for p in te} - set(classes):
        raise RuntimeError("probe train split is missing classes present in test")
    ytr = torch.tensor([classes.index(p.parent.name) for p in tr])
    yte = torch.tensor([classes.index(p.parent.name) for p in te])
    config = {"task": "claim_k4_sink_removal", "sink": SINK, "dataset": DATASET, "n_tr": N_TR, "n_te": N_TE,
              "primary_k": PRIMARY_K, "device": device, "torch": torch.__version__, "timm": timm.__version__,
              "commit": os.environ.get("ORX_COMMIT"),
              "te_sha1": hashlib.sha1("\n".join(p.name for p in te).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    failed = {}
    for key, (r0, c0) in SINK.items():
        t0 = time.time()
        try:
            pr = Probe(key, device)
            g = pr.g
            sink, centre = r0 * g + c0, (g // 2) * g + g // 2
            pe = pr.model.pos_embed.data
            orig = pe.clone()
            print(f"\n== {key} grid={g} sink={(r0, c0)} centre={(g // 2, g // 2)} layers={pr.L}", flush=True)
            for variant in ("clean", "swap", "kill"):
                pe.copy_(orig)
                if variant == "swap":
                    pe[0, sink], pe[0, centre] = orig[0, centre], orig[0, sink]
                elif variant == "kill":
                    pe[0, sink] = orig[0].mean(0)
                ftr, _ = run(pr, tr, sink, False)
                fte, st = run(pr, te, sink, True)
                acc = linear_probe(ftr, ytr, fte, yte)
                print(f"  [{variant}] acc={acc:.4f} outliers/img max={max(st['outliers']):.2f} "
                      f"last={st['outliers'][-1]:.2f} attn_top max={max(st['attn_top']):.3f}", flush=True)
                print("    layer: outliers/img top_is_sink | attn_sink attn_top", flush=True)
                for l in range(pr.L):
                    print(f"    {l:2d}: {st['outliers'][l]:.2f} {st['top_is_sink'][l]:.2f} | "
                          f"{st['attn_sink'][l]:.3f} {st['attn_top'][l]:.3f}", flush=True)
                print("RESULT_JSON " + json.dumps({"key": key, "variant": variant, "acc": acc, **st}), flush=True)
            pe.copy_(orig)
            for m in pr.model.modules():  # break hook ref cycles before the next model
                m._forward_hooks.clear()
                m._forward_pre_hooks.clear()
            del pr
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"{key} done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep the other model's results
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failed[key]}", flush=True)
    print("failed:", json.dumps(failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
