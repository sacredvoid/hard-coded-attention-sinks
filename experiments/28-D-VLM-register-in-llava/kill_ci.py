"""Claim K4-CI: confidence intervals for the cost of killing a positional sink.

SigLIP2 So400m/16@256 (sink (11,0), run K) and AIMv2 L/14 (sink (0,0), run G2). Variants:
clean; kill = the sink patch's position embedding replaced by the mean position embedding.
Imagewoof linear probe on mean-pooled final patch tokens. 3 probe seeds, each with its own
1000-image train subset; one fixed 1000-image val set. Per-image correctness kept, so the
kill - clean difference gets a paired bootstrap 95% CI per seed and pooled over seeds.
"""

import hashlib
import json
import os
import random
import time

import numpy as np
import timm
import torch
from PIL import Image

from measure_outliers import MODELS, image_paths
from probe import linear_probe_preds

MODELS["siglip2_so400m16_256"] = "vit_so400m_patch16_siglip_256.v2_webli"
MODELS["aimv2_l14"] = "aimv2_large_patch14_224.apple_pt"  # not in this branch's (run K) MODELS list
SINK = {"siglip2_so400m16_256": (11, 0), "aimv2_l14": (0, 0)}
DATASET, N_TR, N_TE, SEEDS, BS = "imagewoof2-320", 1000, 1000, (0, 1, 2), 16


@torch.no_grad()
def feats(model, tf, paths, device):
    out = []
    for s in range(0, len(paths), BS):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[s:s + BS]]).to(device)
        out.append(model.forward_features(x)[:, model.num_prefix_tokens:].float().mean(1).cpu())
        if (s // BS) % 32 == 31:  # progress: first attempt a13f1597 was killed silently mid-pass
            print(f"    feats {s + len(x)}/{len(paths)}", flush=True)
    return torch.cat(out)


def boot_paired(diffs, n=4000, seed=0):
    """diffs: [seeds, images] per-image (kill - clean) correctness. Resample images jointly."""
    d = np.asarray(diffs, float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, d.shape[1], (n, d.shape[1]))
    per_seed = [(float(d[s].mean()), *np.percentile(d[s][idx].mean(1), [2.5, 97.5]).tolist()) for s in range(len(d))]
    pooled = d[:, idx].mean(axis=(0, 2))
    return per_seed, (float(d.mean()), *np.percentile(pooled, [2.5, 97.5]).tolist())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    te = image_paths(N_TE, 0, "val", DATASET)
    trs = {s: image_paths(N_TR, s, "train", DATASET) for s in SEEDS}
    union = sorted({p for v in trs.values() for p in v})
    classes = sorted({p.parent.name for p in union})
    lab = lambda ps: torch.tensor([classes.index(p.parent.name) for p in ps])
    config = {"task": "claim_k4_ci", "sink": SINK, "dataset": DATASET, "n_tr": N_TR, "n_te": N_TE, "seeds": SEEDS,
              "n_train_union": len(union), "device": device, "torch": torch.__version__, "timm": timm.__version__,
              "commit": os.environ.get("ORX_COMMIT"),
              "te_sha1": hashlib.sha1("\n".join(p.name for p in te).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    failed = {}
    for key, (r0, c0) in SINK.items():
        t0 = time.time()
        try:
            model = timm.create_model(MODELS[key], pretrained=True, num_classes=0).eval().to(device)
            tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
            g = model.patch_embed.grid_size[0]
            npt = model.num_prefix_tokens
            pe = model.pos_embed.data
            orig = pe.clone()
            if npt != 0 or pe.shape[1] != g * g:
                raise RuntimeError(f"{key}: expected a patch-only pos_embed, got npt={npt} rows={pe.shape[1]}")
            sink = r0 * g + c0
            res = {}
            for variant in ("clean", "kill"):
                pe.copy_(orig)
                if variant == "kill":
                    pe[0, sink] = orig[0].mean(0)  # same kill as runs G2 / K4
                fu = dict(zip(union, feats(model, tf, union, device)))
                fte = feats(model, tf, te, device)
                res[variant] = []
                for s in SEEDS:
                    ftr = torch.stack([fu[p] for p in trs[s]])
                    pred = linear_probe_preds(ftr, lab(trs[s]), fte, lab(te))
                    res[variant].append((pred == lab(te)).int().tolist())
                accs = [float(np.mean(c)) for c in res[variant]]
                print(f"  {key} {variant}: acc per seed {[round(a * 100, 2) for a in accs]}", flush=True)
            pe.copy_(orig)
            diffs = np.array(res["kill"]) - np.array(res["clean"])
            per_seed, pooled = boot_paired(diffs)
            line = {"key": key, "acc_clean": [float(np.mean(c)) for c in res["clean"]],
                    "acc_kill": [float(np.mean(c)) for c in res["kill"]], "diff_per_seed": per_seed, "diff_pooled": pooled}
            print("RESULT_JSON " + json.dumps(line), flush=True)
            print(f"{key}: kill - clean = {pooled[0] * 100:+.2f} pts [{pooled[1] * 100:+.2f}, {pooled[2] * 100:+.2f}] "
                  f"(pooled over {len(SEEDS)} seeds; {time.time() - t0:.0f}s)", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # keep the other model's result
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failed[key]}", flush=True)
    print("failed:", json.dumps(failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
