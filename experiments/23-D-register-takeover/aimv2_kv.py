"""Claim A2: does killed AIMv2 just need the sink's key/value in block 0? And where is the cliff?

Killed model = pos_embed[(0,0)] replaced by the mean pos embed (A1: -30 pts on Imagewoof).
Conditions (Imagewoof probe as in A1, paired bootstrap vs clean):
  alpha{a}   pos embed blended a of the way toward the mean (A1: 0.75 harmless, 1.0 collapses)
  kv0@0      killed, but in block 0 token 0's key and value (qkv channels C..3C) are the clean
             model's; token 0's own hidden state stays killed
  kv0@all    same at every block
  q0@0       control: only token 0's query replaced in block 0
"""

import hashlib
import json
import os
import time

import timm
import torch
from PIL import Image

from aimv2_collapse import BS, DATASET, N_TE, N_TR

# A3: same test on SigLIP2 So400m/16@256, whose sink is patch (11,0) (runs K, K2, K4-CI)
NAME = "vit_so400m_patch16_siglip_256.v2_webli"
SINK = 11 * 16 + 0
from kill_ci import boot_paired
from measure_outliers import image_paths
from probe import linear_probe_preds

ALPHAS = ()  # A3: cliff already mapped for AIMv2 in A2


class KV:
    def __init__(self, device):
        self.device = device
        self.m = timm.create_model(NAME, pretrained=True, num_classes=0).eval().to(device)
        self.tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=self.m))
        self.pe = self.m.pos_embed.data
        self.orig = self.pe.clone()
        self.C = self.m.embed_dim
        self.mode, self.rec = None, {}
        if self.m.num_prefix_tokens != 0 or self.pe.shape[1] != self.m.patch_embed.num_patches:
            raise RuntimeError("expected a patch-only token sequence")
        self.keep = torch.ones(self.pe.shape[1], dtype=torch.bool)
        self.keep[SINK] = False
        for l, b in enumerate(self.m.blocks):
            b.attn.qkv.register_forward_hook(self._hook(l))

    def _hook(self, l):
        def fn(m, i, o):
            if self.mode is None:
                return None
            if self.mode == "record":
                self.rec[l] = o[:, SINK].detach().clone()
                return None
            part, layers = self.mode
            if l not in layers:
                return None
            o = o.clone()
            sl = slice(self.C, 3 * self.C) if part == "kv" else slice(0, self.C)
            o[:, SINK, sl] = self.rec[l][:, sl]
            return o
        return fn

    def set_pe(self, a):
        self.pe.copy_(self.orig)
        if a:
            self.pe[0, SINK] = (1 - a) * self.orig[0, SINK] + a * self.orig[0].mean(0)

    @torch.no_grad()
    def feats(self, paths, cond):
        out = []
        L = len(self.m.blocks)
        for s in range(0, len(paths), BS):
            x = torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths[s:s + BS]]).to(self.device)
            if cond.startswith(("kv0", "q0")):
                self.set_pe(0)
                self.mode = "record"
                self.m.forward_features(x)
                self.set_pe(1.0)
                where = cond.split("@")[1]
                self.mode = (cond[:2] if cond.startswith("kv") else "q", set(range(L)) if where == "all" else {0})
            else:
                self.mode = None
                self.set_pe({"clean": 0.0, "kill": 1.0}.get(cond) if cond in ("clean", "kill") else float(cond[5:]))
            out.append(self.m.forward_features(x)[:, self.keep.to(x.device)].float().mean(1).cpu())  # all but the sink
            self.mode = None
        self.set_pe(0)
        return torch.cat(out)


def selftest():
    global SINK
    saved, SINK = SINK, 0
    k = KV.__new__(KV)
    k.C, k.rec = 2, {0: torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])}
    o = torch.zeros(1, 2, 6)
    k.mode = ("kv", {0})
    assert k._hook(0)(None, None, o)[0, 0].tolist() == [0, 0, 3, 4, 5, 6]
    k.mode = ("q", {0})
    r = k._hook(0)(None, None, o)
    assert r[0, 0].tolist() == [1, 2, 0, 0, 0, 0] and r[0, 1].tolist() == [0] * 6
    assert k._hook(1)(None, None, o) is None
    SINK = saved


def main():
    selftest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = image_paths(N_TR, 0, "train", DATASET), image_paths(N_TE, 0, "val", DATASET)
    classes = sorted({p.parent.name for p in tr})
    ytr = torch.tensor([classes.index(p.parent.name) for p in tr])
    yte = torch.tensor([classes.index(p.parent.name) for p in te])
    conds = ["clean", "kill"] + [f"alpha{a}" for a in ALPHAS] + ["kv0@0", "kv0@all", "q0@0"]
    print("config:", json.dumps({"task": "claim_a3_siglip2_kv", "sink": SINK, "model": NAME, "conds": conds, "device": device,
                                 "commit": os.environ.get("ORX_COMMIT"),
                                 "te_sha1": hashlib.sha1("\n".join(p.name for p in te).encode()).hexdigest()[:12]}),
          flush=True)
    t0 = time.time()
    kv = KV(device)
    correct = {}
    for c in conds:
        pred = linear_probe_preds(kv.feats(tr, c), ytr, kv.feats(te, c), yte)
        correct[c] = (pred == yte).int().numpy()
        line = {"cond": c, "acc": float(correct[c].mean())}
        if c != "clean":
            line["vs_clean"] = boot_paired((correct[c] - correct["clean"])[None])[1]
        print("RESULT_JSON " + json.dumps(line), flush=True)
        vs = line.get("vs_clean")
        print(f"  {c:>9}: acc={line['acc'] * 100:.1f}" +
              (f" vs clean {vs[0] * 100:+.1f} [{vs[1] * 100:+.1f}, {vs[2] * 100:+.1f}]" if vs else "") +
              f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
