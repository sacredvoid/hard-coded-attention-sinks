"""Claim A1: why does AIMv2 collapse when its positional sink is killed?

AIMv2 L/14, sink at patch (0,0) = token 0 (no prefix tokens). Imagewoof linear probe on
mean-pooled final patch tokens (1000 train / 1000 val, seed 0); paired bootstrap vs clean.
Conditions:
  clean / kill            pos_embed[0] -> mean pos embed (as in G2, K4-CI)
  alpha{a}                pos_embed[0] -> (1-a) * orig + a * mean   (dose-response)
  patch@{l}               killed model, but token 0 at block l's output replaced by the clean
                          model's token 0 (same image, same batch): which block's sink matters?
  massive@{l}             same, but only the clean token's top-5 |activation| channels
Divergence: per block, cosine between clean and killed mean patch tokens (excluding token 0)
and token-0 norm / median patch norm in each model.
"""

import hashlib
import json
import os
import time

import numpy as np
import timm
import torch
from PIL import Image

from kill_ci import DATASET, boot_paired
from measure_outliers import image_paths
from probe import linear_probe_preds

NAME = "aimv2_large_patch14_224.apple_pt"
N_TR, N_TE, BS = 1000, 1000, 32
ALPHAS = (0.25, 0.5, 0.75)
PATCH_LAYERS = (0, 1, 2, 4, 8, 12)
MASSIVE_LAYERS = (1, 4)
N_TOP = 5


class Runner:
    def __init__(self, device):
        self.device = device
        self.m = timm.create_model(NAME, pretrained=True, num_classes=0).eval().to(device)
        self.tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=self.m))
        if self.m.num_prefix_tokens != 0:
            raise RuntimeError("expected no prefix tokens: sink must be token 0")
        self.pe = self.m.pos_embed.data
        self.orig = self.pe.clone()
        self.mean_row = self.orig[0].mean(0)
        self.L = len(self.m.blocks)
        self.mode = None  # None | ("record",) | ("patch", l, channels-or-None)
        self.rec = {}
        self.trace = None
        for l, b in enumerate(self.m.blocks):
            b.register_forward_hook(self._hook(l))

    def _hook(self, l):
        def fn(m, i, o):
            if self.trace is not None:
                self.trace[l].append(o.float())
            if self.mode is None:
                return None
            if self.mode[0] == "record":
                self.rec[l] = o[:, 0].detach().clone()
                return None
            _, pl, top = self.mode
            if l != pl:
                return None
            o = o.clone()
            if top is None:
                o[:, 0] = self.rec[l]
            else:
                ch = self.rec[l].abs().mean(0).topk(top).indices
                o[:, 0, ch] = self.rec[l][:, ch]
            return o
        return fn

    def set_pe(self, alpha):
        self.pe.copy_(self.orig)
        if alpha:
            self.pe[0, 0] = (1 - alpha) * self.orig[0, 0] + alpha * self.mean_row

    @torch.no_grad()
    def feats(self, paths, cond):
        out = []
        for s in range(0, len(paths), BS):
            x = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths[s:s + BS]]).to(self.device)
            if cond.startswith(("patch@", "massive@")):
                l = int(cond.split("@")[1])
                self.set_pe(0)
                self.mode = ("record",)
                self.m.forward_features(x)  # clean pass: record token 0 at every block
                self.set_pe(1.0)
                self.mode = ("patch", l, N_TOP if cond.startswith("massive") else None)
            else:
                self.mode = None
                self.set_pe({"clean": 0, "kill": 1.0}.get(cond, float(cond[5:]) if cond.startswith("alpha") else 0))
            y = self.m.forward_features(x)
            out.append(y[:, 1:].float().mean(1).cpu())  # patches except the sink token
            self.mode = None
        self.set_pe(0)
        return torch.cat(out)

    @torch.no_grad()
    def divergence(self, paths):
        res = {"cos_mean_patch": [], "tok0_ratio_clean": [], "tok0_ratio_kill": []}
        x = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths]).to(self.device)
        runs = {}
        for name, a in (("clean", 0), ("kill", 1.0)):
            self.set_pe(a)
            self.trace = [[] for _ in range(self.L)]
            self.m.forward_features(x)
            runs[name] = [torch.cat(t) for t in self.trace]
            self.trace = None
        self.set_pe(0)
        for l in range(self.L):
            c, k = runs["clean"][l], runs["kill"][l]
            res["cos_mean_patch"].append(float(torch.nn.functional.cosine_similarity(
                c[:, 1:].mean(1), k[:, 1:].mean(1), dim=-1).mean()))
            for nm, h in (("clean", c), ("kill", k)):
                n = h.norm(dim=-1)
                res[f"tok0_ratio_{nm}"].append(float((n[:, 0] / n[:, 1:].median(1).values).mean()))
        return res


def selftest():
    # the patch hook must only touch token 0 (and only the chosen channels)
    r = Runner.__new__(Runner)
    r.trace, r.rec = None, {3: torch.tensor([[9.0, 8.0, 0.0]])}
    r.mode = ("patch", 3, 1)
    o = torch.zeros(1, 2, 3)
    new = r._hook(3)(None, None, o)
    assert new[0, 0].tolist() == [9.0, 0.0, 0.0] and new[0, 1].tolist() == [0.0, 0.0, 0.0]
    r.mode = ("patch", 3, None)
    assert r._hook(3)(None, None, o)[0, 0].tolist() == [9.0, 8.0, 0.0]
    assert r._hook(2)(None, None, o) is None


def main():
    selftest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = image_paths(N_TR, 0, "train", DATASET), image_paths(N_TE, 0, "val", DATASET)
    classes = sorted({p.parent.name for p in tr})
    ytr = torch.tensor([classes.index(p.parent.name) for p in tr])
    yte = torch.tensor([classes.index(p.parent.name) for p in te])
    conds = (["clean", "kill"] + [f"alpha{a}" for a in ALPHAS] + [f"patch@{l}" for l in PATCH_LAYERS]
             + [f"massive@{l}" for l in MASSIVE_LAYERS])
    print("config:", json.dumps({"task": "claim_a1_aimv2_collapse", "model": NAME, "dataset": DATASET,
                                 "conds": conds, "n_top": N_TOP, "device": device, "commit": os.environ.get("ORX_COMMIT"),
                                 "te_sha1": hashlib.sha1("\n".join(p.name for p in te).encode()).hexdigest()[:12]}),
          flush=True)
    t0 = time.time()
    run = Runner(device)
    div = run.divergence(te[:32])
    print("divergence per block (cos clean/kill mean patch | tok0 ratio clean | tok0 ratio kill):", flush=True)
    for l in range(run.L):
        print(f"  {l:2d}: {div['cos_mean_patch'][l]:.3f} | {div['tok0_ratio_clean'][l]:.1f} | "
              f"{div['tok0_ratio_kill'][l]:.1f}", flush=True)
    print("RESULT_JSON " + json.dumps({"divergence": div}), flush=True)
    correct = {}
    for c in conds:
        pred = linear_probe_preds(run.feats(tr, c), ytr, run.feats(te, c), yte)
        correct[c] = (pred == yte).int().numpy()
        line = {"cond": c, "acc": float(correct[c].mean())}
        if c != "clean":
            _, pooled = boot_paired((correct[c] - correct["clean"])[None])
            line["vs_clean"] = pooled
        print("RESULT_JSON " + json.dumps(line), flush=True)
        print(f"  {c:>12}: acc={line['acc'] * 100:.1f}" + (f" vs clean {pooled[0] * 100:+.1f} "
              f"[{pooled[1] * 100:+.1f}, {pooled[2] * 100:+.1f}]" if c != "clean" else "") +
              f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
