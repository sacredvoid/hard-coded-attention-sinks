"""Claim C: do outlier patch tokens carry global (image-level) information?

At layers where outliers exist, collect per image: the top-norm patch token (an outlier
when its norm > 4x the image's median patch norm), a random normal patch token
(norm < 2x median), the mean of patch tokens, and CLS if the model has one. Fit linear
probes for Imagenette class on each, using only images that have an outlier at that layer
so every feature is compared on the same images. Outlier >> normal patch means outliers
store image-level information instead of local content (Darcet et al., DINOv2).
"""

import hashlib
import json
import os
import time

import timm
import torch
from PIL import Image

from measure_outliers import MODELS, PRIMARY_K, image_paths
from probe import linear_probe

# Layers chosen from the outlier-positions run: emergence/plateau and last block.
# AIMv2 layer 12 is its fixed (0,0) sink; layer 23 has its late content outliers.
LAYERS = {"clip_l14": [13, 23], "dinov2_l14": [19], "siglip2_l16": [15, 23], "aimv2_l14": [12, 23],
          # clean encoders: no patch outliers, so probe their prefix tokens (registers / CLS) instead
          "dinov2_l14_reg4": [19, 23], "dinov3_l16": [12, 23], "pe_core_l14": [12, 23]}
DATASET = "imagewoof2-320"  # 10 dog breeds: harder than Imagenette, avoids the probe ceiling
NORMAL_MAX = 2.0


def label_of(p) -> str:
    return p.parent.name


@torch.no_grad()
def collect(key, paths, device, bs, seed):
    model = timm.create_model(MODELS[key], pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    npt = int(getattr(model, "num_prefix_tokens", 0))
    gen = torch.Generator(device=device).manual_seed(seed)
    feats = {l: {k: [] for k in ("outlier", "normal", "mean", "ratio", "pos")} for l in LAYERS[key]}

    def hook(layer):
        def fn(m, i, o):
            o = o.float()
            pt = o[:, npt:]
            n = pt.norm(dim=-1)
            ratio = n / n.median(dim=1, keepdim=True).values
            ar = torch.arange(len(o), device=o.device)
            top = ratio.argmax(1)
            normal_w = (ratio < NORMAL_MAX).float() + 1e-9  # eps: never all-zero weights
            rnd = torch.multinomial(normal_w, 1, generator=gen).squeeze(1)
            f = feats[layer]
            f["outlier"].append(pt[ar, top].cpu())
            f["normal"].append(pt[ar, rnd].cpu())
            f["mean"].append(pt.mean(1).cpu())
            for j in range(npt):
                f.setdefault(f"prefix{j}", []).append(o[:, j].cpu())
            f["ratio"].append(ratio.max(1).values.cpu())
            f["pos"].append(top.cpu())
        return fn

    hooks = [model.blocks[l].register_forward_hook(hook(l)) for l in LAYERS[key]]
    for s in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[s:s + bs]]).to(device)
        model.forward_features(x)
    for h in hooks:
        h.remove()
    del model
    return {l: {k: torch.cat(v) for k, v in f.items() if v} for l, f in feats.items()}


def main():
    n_train, n_test, bs, seed = 2000, 1000, 16, 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = image_paths(n_train, seed, "train", DATASET), image_paths(n_test, seed, "val", DATASET)
    classes = sorted({label_of(p) for p in tr})
    ytr = torch.tensor([classes.index(label_of(p)) for p in tr])
    yte = torch.tensor([classes.index(label_of(p)) for p in te])
    config = {"task": "claim_c2_global_info", "dataset": DATASET, "n_train": n_train, "n_test": n_test, "seed": seed,
              "device": device, "layers": LAYERS, "primary_k": PRIMARY_K, "normal_max": NORMAL_MAX,
              "torch": torch.__version__, "timm": timm.__version__, "commit": os.environ.get("ORX_COMMIT"),
              "test_sha1": hashlib.sha1("\n".join(p.name for p in te).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    results, failed = {}, {}
    for key in LAYERS:
        t0 = time.time()
        try:
            ftr, fte = collect(key, tr, device, bs, seed), collect(key, te, device, bs, seed + 1)
            results[key] = {}
            for l in LAYERS[key]:
                mtr, mte = ftr[l]["ratio"] > PRIMARY_K, fte[l]["ratio"] > PRIMARY_K
                pos = torch.cat([ftr[l]["pos"][mtr], fte[l]["pos"][mte]])
                top_share = float(pos.bincount().max() / len(pos)) if len(pos) else None
                r = {"n_train": int(mtr.sum()), "n_test": int(mte.sum()),
                     "top_pos": int(pos.bincount().argmax()) if len(pos) else None, "top_pos_share": top_share}
                feats = [k for k in ftr[l] if k not in ("ratio", "pos")]
                if r["n_train"] < 100 or r["n_test"] < 50:  # no patch outliers: all images, no outlier probe
                    mtr, mte = torch.ones_like(mtr), torch.ones_like(mte)
                    r["subset"] = "all images (too few with a patch outlier)"
                    feats.remove("outlier")
                for k in feats:
                    r[f"acc_{k}"] = linear_probe(ftr[l][k][mtr], ytr[mtr], fte[l][k][mte], yte[mte])
                # chance-level reference: same normal tokens, labels shuffled
                perm = torch.randperm(int(mtr.sum()), generator=torch.Generator().manual_seed(0))
                r["acc_normal_shuffled"] = linear_probe(ftr[l]["normal"][mtr], ytr[mtr][perm],
                                                        fte[l]["normal"][mte], yte[mte])
                results[key][l] = r
                print(f"{key} L{l}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                 for k, v in r.items()), flush=True)
            print(f"{key} done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep other encoders' results
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failed[key]}", flush=True)
    print("\n=== SUMMARY_JSON_BEGIN ===")
    print(json.dumps({"config": config, "results": results, "failed": failed}))
    print("=== SUMMARY_JSON_END ===")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
