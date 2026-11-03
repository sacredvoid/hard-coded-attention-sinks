"""Claim D: can one constant extra token take over a hard-coded sink at test time?

A2/A3: the sink works only as a key/value target; H: the sink token is nearly input-invariant.
So kill the position-embedding sink (mean pos embed at the sink patch) and append ONE constant
token before block 0: the mean, over calibration images, of the clean block-0 input at the sink
patch. No training. Conditions:
  clean | kill | kill+reg | clean+reg | kill+zero (zero vector appended: control)
Imagewoof probe (1000/1000), features = mean of patches excluding the sink patch and the
register; paired bootstrap vs clean. Also the norm ratio (vs median patch) of the sink patch and
of the register at the middle and last block.
"""

import hashlib
import json
import os
import time

import timm
import torch
from PIL import Image

from aimv2_collapse import BS, DATASET, N_TE, N_TR
from kill_ci import boot_paired
from measure_outliers import image_paths
from probe import linear_probe_preds

MODELS = {"aimv2_l14": ("aimv2_large_patch14_224.apple_pt", 0),
          "siglip2_so400m16_256": ("vit_so400m_patch16_siglip_256.v2_webli", 11 * 16 + 0)}
N_CAL = 200
CONDS = ["clean", "kill", "kill+reg", "clean+reg", "kill+zero"]


class Takeover:
    def __init__(self, name, sink, device):
        self.device, self.sink = device, sink
        self.m = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)
        self.tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=self.m))
        self.pe = self.m.pos_embed.data
        if self.m.num_prefix_tokens != 0 or self.pe.shape[1] != self.m.patch_embed.num_patches:
            raise RuntimeError("expected a patch-only token sequence")
        self.orig = self.pe.clone()
        self.N = self.pe.shape[1]
        self.reg = None  # [D] vector appended before block 0, or None
        self.record = None  # list collecting clean block-0 inputs at the sink
        self.ratios = {}
        L = len(self.m.blocks)
        self.watch = {L // 2: "mid", L - 1: "last"}
        self.m.blocks[0].register_forward_pre_hook(self._pre)
        for l in self.watch:
            self.m.blocks[l].register_forward_hook(self._post(l))

    def _pre(self, m, args):
        x = args[0]
        if self.record is not None:
            self.record.append(x[:, self.sink].float().sum(0))
        if self.reg is None:
            return None
        r = self.reg.to(x.dtype).expand(x.shape[0], 1, -1)
        return (torch.cat([x, r], 1),) + args[1:]

    def _post(self, l):
        def fn(m, i, o):
            n = o.float().norm(dim=-1)
            med = n[:, :self.N].median(1).values
            d = self.ratios.setdefault(self.watch[l], {"sink": [], "reg": []})
            d["sink"].append((n[:, self.sink] / med).cpu())
            if o.shape[1] > self.N:
                d["reg"].append((n[:, self.N] / med).cpu())
        return fn

    def batches(self, paths):
        for s in range(0, len(paths), BS):
            yield torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths[s:s + BS]]).to(self.device)

    @torch.no_grad()
    def calibrate(self, paths):
        self.pe.copy_(self.orig)
        self.reg, self.record = None, []
        n = 0
        for x in self.batches(paths):
            self.m.forward_features(x)
            n += len(x)
        self.reg_mean = torch.stack(self.record).sum(0) / n
        self.record = None

    @torch.no_grad()
    def feats(self, paths, cond):
        self.pe.copy_(self.orig)
        if cond.startswith("kill"):
            self.pe[0, self.sink] = self.orig[0].mean(0)
        self.reg = {"reg": self.reg_mean, "zero": torch.zeros_like(self.reg_mean)}.get(cond.partition("+")[2])
        keep = torch.ones(self.N, dtype=torch.bool, device=self.device)
        keep[self.sink] = False
        self.ratios = {}
        out = []
        for x in self.batches(paths):
            y = self.m.forward_features(x)[:, :self.N]  # drop the appended register
            out.append(y[:, keep].float().mean(1).cpu())
        self.reg = None
        self.pe.copy_(self.orig)
        rat = {w: {k: float(torch.cat(v).mean()) for k, v in d.items() if v} for w, d in self.ratios.items()}
        return torch.cat(out), rat


def selftest():
    t = Takeover.__new__(Takeover)
    t.sink, t.record, t.reg = 1, None, torch.tensor([7.0, 7.0])
    x = torch.zeros(2, 3, 2)
    y = t._pre(None, (x,))[0]
    assert y.shape == (2, 4, 2) and y[:, -1].tolist() == [[7.0, 7.0]] * 2 and y[:, :3].abs().sum() == 0
    t.reg, t.record = None, []
    assert t._pre(None, (x + 1,)) is None and t.record[0].tolist() == [2.0, 2.0]


def main():
    selftest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = image_paths(N_TR, 0, "train", DATASET), image_paths(N_TE, 0, "val", DATASET)
    cal = image_paths(N_TR + N_CAL, 1, "train", DATASET)
    tr_set = set(tr)
    cal = [p for p in cal if p not in tr_set][:N_CAL]  # calibration images disjoint from probe train
    classes = sorted({p.parent.name for p in tr})
    ytr = torch.tensor([classes.index(p.parent.name) for p in tr])
    yte = torch.tensor([classes.index(p.parent.name) for p in te])
    print("config:", json.dumps({"task": "claim_d_register_takeover", "models": MODELS, "conds": CONDS,
                                 "n_cal": len(cal), "device": device, "commit": os.environ.get("ORX_COMMIT"),
                                 "te_sha1": hashlib.sha1("\n".join(p.name for p in te).encode()).hexdigest()[:12]}),
          flush=True)
    failed = {}
    for key, (name, sink) in MODELS.items():
        t0 = time.time()
        try:
            t = Takeover(name, sink, device)
            t.calibrate(cal)
            correct = {}
            for c in CONDS:
                ftr, _ = t.feats(tr, c)
                fte, rat = t.feats(te, c)
                correct[c] = (linear_probe_preds(ftr, ytr, fte, yte) == yte).int().numpy()
                line = {"key": key, "cond": c, "acc": float(correct[c].mean()), "norm_ratio": rat}
                if c != "clean":
                    line["vs_clean"] = boot_paired((correct[c] - correct["clean"])[None])[1]
                print("RESULT_JSON " + json.dumps(line), flush=True)
                vs = line.get("vs_clean")
                print(f"  {key} {c:>10}: acc={line['acc'] * 100:.1f}" +
                      (f" vs clean {vs[0] * 100:+.1f} [{vs[1] * 100:+.1f}, {vs[2] * 100:+.1f}]" if vs else "") +
                      f" ratios={json.dumps({w: {k: round(v, 1) for k, v in d.items()} for w, d in rat.items()})}"
                      f" ({time.time() - t0:.0f}s)", flush=True)
            del t
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # keep the other model's results
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failed[key]}", flush=True)
    print("failed:", json.dumps(failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
