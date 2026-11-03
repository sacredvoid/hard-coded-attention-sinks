"""Claim H: remove each encoder's attention sink. Does a new sink appear? Does it matter?

Mean-ablation at every block output (means from a clean calibration pass, so replacements
stay in-distribution, cf. arXiv 2604.14433):
  ablate_outliers  patch tokens with norm > K x median -> that layer's mean normal patch
  ablate_reg{j}    prefix token j -> its per-layer mean;  ablate_all_regs: every register
  ablate_cls       CLS -> its per-layer mean
  mean_pe00        AIMv2 only: (0,0) position embedding -> mean position embedding (run G2)
Per layer: outliers that EMERGE at the block output (counted before ablation), attention mass
on the most-attended key and whether that key is a prefix token, prefix masses. Plus an
Imagenette linear probe on mean-pooled final patch tokens.
"""

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
from sink_mechanism import SINK, free, massive_channels

# G2 measured channels at token idx+1 for the register encoders (off by one); redo at the true sink.
TRUE_PREFIX_SINK = {"dinov2_l14_reg4": ("prefix", 20, 1), "dinov3_l16": ("prefix", 20, 2)}

N_CAL, N_TR, N_TE, SEED = 100, 500, 500, 0
CONFIGS = {"clip_l14": ["clean", "ablate_outliers"], "dinov2_l14": ["clean", "ablate_outliers"],
           "dinov2_l14_reg4": ["clean", "ablate_reg1", "ablate_all_regs"],  # sink = token 1 (run G attention)
           "siglip2_l16": ["clean", "ablate_outliers"],
           "dinov3_l16": ["clean", "ablate_reg2", "ablate_all_regs"],  # sink = token 2 (run G attention)
           "pe_core_l14": ["clean", "ablate_cls"], "aimv2_l14": ["clean", "mean_pe00", "ablate_outliers"]}


class Ablator:
    def __init__(self, pr):
        self.pr, self.mode, self.L, self.npt = pr, "clean", pr.L, pr.npt
        self.mean_patch = [None] * self.L
        self.mean_prefix = [None] * self.L
        self.calibrating = False
        self.cal_n = 0
        self.emerged = [0.0] * self.L
        self.top_is_prefix = [0.0] * self.L
        self.orig_pe = pr.model.pos_embed.data.clone() if getattr(pr.model, "pos_embed", None) is not None else None
        for l, b in enumerate(pr.model.blocks):
            b.register_forward_hook(self._out(l))
            b.attn.attn_drop.register_forward_hook(self._att(l))

    def set_mode(self, mode):
        self.mode = mode
        if self.orig_pe is not None:
            self.pr.model.pos_embed.data.copy_(self.orig_pe)
            if mode == "mean_pe00":
                self.pr.model.pos_embed.data[0, 0] = self.orig_pe[0].mean(0)
        self.emerged = [0.0] * self.L
        self.top_is_prefix = [0.0] * self.L

    def _out(self, l):
        def fn(m, i, o):
            npt = self.npt
            n = o[:, npt:].float().norm(dim=-1)
            ratio = n / n.median(dim=1, keepdim=True).values
            out = ratio > PRIMARY_K
            if self.calibrating:
                normal = (ratio < 2).unsqueeze(-1).float()
                s = (o[:, npt:].float() * normal).sum((0, 1)) / normal.sum().clamp_min(1)
                p = o[:, :npt].float().mean(0)
                k = self.cal_n
                self.mean_patch[l] = s if self.mean_patch[l] is None else (self.mean_patch[l] * k + s) / (k + 1)
                if npt:
                    self.mean_prefix[l] = p if self.mean_prefix[l] is None else (self.mean_prefix[l] * k + p) / (k + 1)
                return None
            self.emerged[l] += out.sum().item()
            mode = self.mode
            if mode in ("clean", "mean_pe00"):
                return None
            o = o.clone()
            if mode == "ablate_outliers":
                o[:, npt:][out] = self.mean_patch[l].to(o.dtype)
            elif mode == "ablate_all_regs":
                o[:, npt - 4:npt] = self.mean_prefix[l][npt - 4:npt].to(o.dtype)
            elif mode == "ablate_cls":
                o[:, 0] = self.mean_prefix[l][0].to(o.dtype)
            elif mode.startswith("ablate_reg"):
                j = int(mode[len("ablate_reg"):])
                o[:, j] = self.mean_prefix[l][j].to(o.dtype)
            else:
                raise ValueError(mode)
            return o
        return fn

    def _att(self, l):
        def fn(m, i, a):
            if self.calibrating:
                return
            k = a[:, :, self.npt:].float().mean(1).mean(1)  # [b, T]
            self.top_is_prefix[l] += (k.argmax(1) < self.npt).sum().item()
        return fn


@torch.no_grad()
def run(pr, ab, paths, collect_attn):
    feats = []
    pr.reset()
    n = 0
    for s in range(0, len(paths), pr.bs):
        x = torch.stack([pr.tf(Image.open(p).convert("RGB")) for p in paths[s:s + pr.bs]]).to(pr.device)
        out = pr.model.forward_features(x).float()
        feats.append(out[:, pr.npt:].mean(1).cpu())
        n += len(x)
    stats = None
    if collect_attn:
        a = pr.acc
        stats = {"emerged_outliers": [v / n for v in ab.emerged], "top_key_mass": [v / n for v in a["top_key_mass"]],
                 "top_key_is_prefix": [v / n for v in ab.top_is_prefix],
                 "prefix_mass": [[v / n for v in row] for row in a["prefix_mass"]]}
    return torch.cat(feats), stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = image_paths(N_CAL + N_TR, SEED, "train")
    random.Random(SEED).shuffle(train)  # sorted by class: shuffle before splitting
    cal, tr = train[:N_CAL], train[N_CAL:]
    te = image_paths(N_TE, SEED, "val")
    classes = sorted({p.parent.name for p in tr})
    if {p.parent.name for p in te} - set(classes):
        raise RuntimeError("probe train split is missing classes present in test")
    ytr = torch.tensor([classes.index(p.parent.name) for p in tr])
    yte = torch.tensor([classes.index(p.parent.name) for p in te])
    config = {"task": "claim_h_sink_ablation", "configs": CONFIGS, "n_cal": N_CAL, "n_tr": N_TR, "n_te": N_TE,
              "primary_k": PRIMARY_K, "device": device, "torch": torch.__version__, "timm": timm.__version__,
              "commit": os.environ.get("ORX_COMMIT"),
              "te_sha1": hashlib.sha1("\n".join(p.name for p in te).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    results, failed = {}, {}
    for key, modes in CONFIGS.items():
        t0 = time.time()
        try:
            pr = Probe(key, device)
            ab = Ablator(pr)
            ab.calibrating = True
            for s in range(0, len(cal), pr.bs):
                x = torch.stack([pr.tf(Image.open(p).convert("RGB")) for p in cal[s:s + pr.bs]]).to(device)
                with torch.no_grad():
                    pr.model.forward_features(x)
                ab.cal_n += 1
            ab.calibrating = False
            print(f"\n== {key} prefix={pr.npt}", flush=True)
            res = {}
            for mode in modes:
                ab.set_mode(mode)
                ftr, _ = run(pr, ab, tr, False)
                ab.set_mode(mode)
                fte, st = run(pr, ab, te, True)
                st["acc"] = linear_probe(ftr, ytr, fte, yte)
                res[mode] = st
                last = pr.L - 1
                peak = max(range(pr.L), key=lambda l: st["top_key_mass"][l])
                print(f"  {mode:>16}: acc={st['acc']:.3f} emerged/img max={max(st['emerged_outliers']):.2f} "
                      f"last={st['emerged_outliers'][last]:.2f} top_key_mass peak={st['top_key_mass'][peak]:.3f}@L{peak} "
                      f"prefix_top_share@peak={st['top_key_is_prefix'][peak]:.2f}", flush=True)
                print("     emerged/layer=" + str([round(v, 2) for v in st["emerged_outliers"]]), flush=True)
                print("     topkey/layer=" + str([round(v, 3) for v in st["top_key_mass"]]), flush=True)
                print("     prefixtop/layer=" + str([round(v, 2) for v in st["top_key_is_prefix"]]), flush=True)
            if key in TRUE_PREFIX_SINK:
                SINK[key] = TRUE_PREFIX_SINK[key]
                ab.set_mode("clean")
                res["massive_true_sink"] = massive_channels(pr, key, te[:200])
                print(f"  massive channels at true sink {SINK[key]}: {json.dumps(res['massive_true_sink'])}", flush=True)
            results[key] = res
            free(pr)
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
