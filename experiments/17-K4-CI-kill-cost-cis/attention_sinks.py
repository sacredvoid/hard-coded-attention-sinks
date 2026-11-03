"""Claim G: are the high-norm tokens attention sinks, and is AIMv2's (0,0) sink positional?

Attention is switched to the non-fused path so softmax probabilities can be read at
attn_drop. Per layer, averaged over images, heads, and patch queries:
  prefix_mass[j]  attention mass on prefix token j (CLS / registers)
  outlier_mass    mass on patch keys that are outliers (norm > K x median) at the block input
  p00_mass        mass on patch (0,0)
  top_key_mass    mass on the single most-attended key (any token), averaged per image
  entropy         attention entropy (nats)
AIMv2 interventions: (0,0) patch set to the dataset mean color ("gray00"), and the image
rolled by 3 patches in both axes ("roll3") so different content lands at (0,0).
"""

import hashlib
import json
import os
import time

import timm
import torch
from PIL import Image

from measure_outliers import MODELS, PRIMARY_K, image_paths

N_IMG, SEED = 300, 0


class Probe:
    def __init__(self, key, device):
        self.device = device
        self.model = timm.create_model(MODELS[key], pretrained=True, num_classes=0).eval().to(device)
        n_attn = 0
        for m in self.model.modules():
            if hasattr(m, "fused_attn"):
                m.fused_attn = False
                n_attn += 1
        self.L = len(self.model.blocks)
        if n_attn < self.L:
            raise RuntimeError(f"{key}: only {n_attn} attention modules expose fused_attn")
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = timm.data.create_transform(**cfg)
        self.npt = int(getattr(self.model, "num_prefix_tokens", 0))
        self.ps = self.model.patch_embed.patch_size[0]
        self.g = cfg["input_size"][1] // self.ps
        self.bs = max(1, min(16, 4096 // self.g ** 2))
        self.in_ratio = None  # ratio of tokens entering the current block
        self.seen = [0] * self.L
        self.reset()
        for l, b in enumerate(self.model.blocks):
            b.register_forward_pre_hook(self._pre(l))
            b.attn.attn_drop.register_forward_hook(self._attn(l))

    def reset(self):
        z = lambda: [0.0] * self.L
        self.acc = {"prefix_mass": [[0.0] * self.npt for _ in range(self.L)], "outlier_mass": z(),
                    "outlier_count": z(), "p00_mass": z(), "top_key_mass": z(), "entropy": z(), "n": 0}
        self.calls = [0] * self.L

    def _pre(self, l):
        def fn(m, args):
            x = args[0]
            n = x[:, self.npt:].float().norm(dim=-1)
            self.in_ratio = n / n.median(dim=1, keepdim=True).values
        return fn

    def _attn(self, l):
        def fn(m, i, a):  # a: [b, heads, T, T] softmax probs
            self.calls[l] += 1
            q = a[:, :, self.npt:].float().mean(1)  # [b, patches, T] head-mean, patch queries
            per_key = q.mean(1)  # [b, T] mean over queries
            out = self.in_ratio > PRIMARY_K  # [b, patches]
            A = self.acc
            for j in range(self.npt):
                A["prefix_mass"][l][j] += per_key[:, j].sum().item()
            A["outlier_mass"][l] += (per_key[:, self.npt:] * out).sum().item()
            A["outlier_count"][l] += out.sum().item()
            A["p00_mass"][l] += per_key[:, self.npt].sum().item()
            A["top_key_mass"][l] += per_key.max(1).values.sum().item()
            ent = -(a[:, :, self.npt:].float().clamp_min(1e-12).log() * a[:, :, self.npt:].float()).sum(-1)
            A["entropy"][l] += ent.mean((1, 2)).sum().item()
        return fn

    @torch.no_grad()
    def run(self, paths, edit=None):
        self.reset()
        for s in range(0, len(paths), self.bs):
            x = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths[s:s + self.bs]]).to(self.device)
            if edit:
                x = edit(x)
            self.model.forward_features(x)
            self.acc["n"] += len(x)
        if any(c == 0 for c in self.calls):
            raise RuntimeError(f"attention hook never fired for layers {[l for l, c in enumerate(self.calls) if c == 0]}")
        n = self.acc.pop("n")
        r = {k: [[v / n for v in row] if isinstance(row, list) else row / n for row in vals] for k, vals in self.acc.items()}
        return r


def gray00(ps):
    def f(x):
        x = x.clone()
        x[..., :ps, :ps] = 0.0  # normalized space: 0 == dataset mean color
        return x
    return f


def roll3(ps):
    return lambda x: x.roll((3 * ps, 3 * ps), dims=(-2, -1))


def show(key, name, r, L):
    print(f"  [{name}] layer: prefix_mass | outlier_mass (count/img) | p00_mass | top_key_mass | entropy", flush=True)
    for l in range(L):
        pm = "[" + ",".join(f"{v:.3f}" for v in r["prefix_mass"][l]) + "]"
        print(f"  {l:2d}: {pm} | {r['outlier_mass'][l]:.3f} ({r['outlier_count'][l]:.2f}) | {r['p00_mass'][l]:.3f} | "
              f"{r['top_key_mass'][l]:.3f} | {r['entropy'][l]:.2f}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = image_paths(N_IMG, SEED)  # val split
    config = {"task": "claim_g_attention_sinks", "n_img": N_IMG, "primary_k": PRIMARY_K, "device": device,
              "torch": torch.__version__, "timm": timm.__version__, "commit": os.environ.get("ORX_COMMIT"),
              "images_sha1": hashlib.sha1("\n".join(p.name for p in paths).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    results, failed = {}, {}
    for key in MODELS:
        t0 = time.time()
        try:
            pr = Probe(key, device)
            print(f"\n== {key} grid={pr.g} prefix={pr.npt}", flush=True)
            res = {"clean": pr.run(paths)}
            show(key, "clean", res["clean"], pr.L)
            if key == "aimv2_l14":
                for name, ed in (("gray00", gray00(pr.ps)), ("roll3", roll3(pr.ps))):
                    res[name] = pr.run(paths, ed)
                    show(key, name, res[name], pr.L)
            results[key] = res
            del pr
            if device == "cuda":
                torch.cuda.empty_cache()
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
