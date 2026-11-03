"""Claim E: do test-time registers (Jiang et al., arXiv 2506.08010) transfer unchanged?

Recipe, applied the same way to every encoder:
1. Calibration (200 train images): at the detection layer DET (first layer where most
   images have a >4x patch outlier), mark outlier positions. For every MLP hidden neuron
   in blocks 0..DET, score = mean activation over those outlier positions.
2. Register neurons = top-K scores across all those layers.
3. Test time: append one zero token to the sequence (the test-time register). In each
   register neuron, write the max activation over the other tokens into the register
   and zero it everywhere else.

Configs: base (no extra token), reg_only (extra token, no shift; control), K in K_SWEEP.
Per config: mean patch-outlier count per image per layer (val), register-token norm
ratio, and linear-probe accuracy (Imagenette) on mean-pooled final patch tokens (and CLS).
"""

import hashlib
import json
import os
import random
import time

import timm
import torch
from PIL import Image

from measure_outliers import MODELS, PRIMARY_K, image_paths
from probe import linear_probe

DET = {"clip_l14": 11, "dinov2_l14": 17, "siglip2_l16": 10, "aimv2_l14": 17}  # from positions run
K_SWEEP = [5, 10, 20, 50]
N_CALIB, N_TRAIN, N_TEST, BS, SEED = 200, 1000, 1000, 16, 0


class Harness:
    def __init__(self, key, device):
        self.key, self.device = key, device
        self.model = timm.create_model(MODELS[key], pretrained=True, num_classes=0).eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = timm.data.create_transform(**cfg)
        self.npt = int(getattr(self.model, "num_prefix_tokens", 0))
        self.L = len(self.model.blocks)
        self.reg = False
        self.neurons: dict[int, torch.Tensor] = {}
        self.calib_mask = None  # [b, T] bool: outlier token positions during calibration
        self.ratio = [None] * self.L
        self.act_sum = [None] * self.L
        self.act_n = 0
        self.model.blocks[0].register_forward_pre_hook(self._add_register)
        for l, blk in enumerate(self.model.blocks):
            if not hasattr(blk.mlp, "norm"):
                raise RuntimeError(f"{key}: block {l} MLP has no .norm hook point")
            blk.mlp.norm.register_forward_hook(self._mlp_hook(l))
            blk.register_forward_hook(self._block_hook(l))

    def _add_register(self, m, args):
        if not self.reg:
            return None
        x = args[0]
        return (torch.cat([x, x.new_zeros(x.shape[0], 1, x.shape[2])], 1),) + args[1:]

    def _mlp_hook(self, l):
        def fn(m, i, out):
            if self.calib_mask is not None and l <= DET[self.key]:
                s = (out.float() * self.calib_mask[..., None]).sum((0, 1))
                self.act_sum[l] = s if self.act_sum[l] is None else self.act_sum[l] + s
            if self.reg and l in self.neurons:
                idx = self.neurons[l]
                out = out.clone()
                v = out[:, :-1, idx].max(dim=1).values
                out[:, :-1, idx] = 0
                out[:, -1, idx] = v
                return out
            return None
        return fn

    def _block_hook(self, l):
        def fn(m, i, o):
            end = o.shape[1] - 1 if self.reg else o.shape[1]
            n = o[:, self.npt:end].float().norm(dim=-1)
            med = n.median(dim=1, keepdim=True).values
            self.ratio[l] = n / med
            if self.reg and l == self.L - 1:
                self.reg_ratio = (o[:, -1].float().norm(dim=-1) / med.squeeze(1))
        return fn

    def batches(self, paths):
        for s in range(0, len(paths), BS):
            yield torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths[s:s + BS]]).to(self.device)

    @torch.no_grad()
    def calibrate(self, paths):
        self.reg, self.neurons = False, {}
        for x in self.batches(paths):
            self.model.forward_features(x)
            m = self.ratio[DET[self.key]] > PRIMARY_K  # [b, patches]
            full = torch.zeros(m.shape[0], m.shape[1] + self.npt, device=m.device)
            full[:, self.npt:] = m.float()
            self.calib_mask = full
            self.model.forward_features(x)
            self.calib_mask = None
            self.act_n += int(m.sum())
        if self.act_n == 0:
            raise RuntimeError(f"{self.key}: no outliers at layer {DET[self.key]} during calibration")
        layers = list(range(DET[self.key] + 1))
        scores = torch.stack([self.act_sum[l] / self.act_n for l in layers])  # [layers, H]
        H = scores.shape[1]
        order = scores.flatten().argsort(descending=True)
        self.ranked = [(layers[int(i) // H], int(i) % H, float(scores.flatten()[i])) for i in order[:max(K_SWEEP)]]

    def set_config(self, cfg):
        self.reg = cfg != "base"
        self.neurons = {}
        if isinstance(cfg, int):
            by_layer: dict[int, list[int]] = {}
            for l, n, _ in self.ranked[:cfg]:
                by_layer.setdefault(l, []).append(n)
            self.neurons = {l: torch.tensor(v, device=self.device) for l, v in by_layer.items()}

    @torch.no_grad()
    def features(self, paths, record=False):
        feats, cls, counts, maxr, regr = [], [], [], [], []
        for x in self.batches(paths):
            out = self.model.forward_features(x).float()
            if self.reg:
                out = out[:, :-1]
            feats.append(out[:, self.npt:].mean(1).cpu())
            if self.npt:
                cls.append(out[:, 0].cpu())
            if record:
                r = torch.stack(self.ratio, 1)  # [b, L, patches]
                counts.append((r > PRIMARY_K).sum(2).float().cpu())
                maxr.append(r.max(2).values.cpu())
                if self.reg:
                    regr.append(self.reg_ratio.cpu())
        res = {"mean": torch.cat(feats), "cls": torch.cat(cls) if cls else None}
        if record:
            res["count"] = torch.cat(counts).mean(0).tolist()
            res["max_ratio"] = torch.cat(maxr).mean(0).tolist()
            res["reg_ratio"] = torch.cat(regr).mean().item() if regr else None
        return res


def selftest():
    """Shift must move a neuron's max onto the register token and zero the rest."""
    h = Harness.__new__(Harness)
    h.key, h.calib_mask, h.reg = "clip_l14", None, True
    h.neurons = {0: torch.tensor([1])}
    out = torch.tensor([[[1.0, 3.0], [2.0, 5.0], [0.0, 0.0]]])  # 2 tokens + register
    new = h._mlp_hook(0)(None, None, out)
    assert new[0, :, 0].tolist() == [1.0, 2.0, 0.0] and new[0, :, 1].tolist() == [0.0, 0.0, 5.0]


def main():
    selftest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    te = image_paths(N_TEST, SEED, "val")
    train = image_paths(N_CALIB + N_TRAIN, SEED, "train")  # sorted by class dir: shuffle before split
    random.Random(SEED).shuffle(train)
    calib, tr = train[:N_CALIB], train[N_CALIB:]
    if {p.parent.name for p in te} - {p.parent.name for p in tr}:
        raise RuntimeError("probe train split is missing classes present in test")
    classes = sorted({p.parent.name for p in tr})
    ytr = torch.tensor([classes.index(p.parent.name) for p in tr])
    yte = torch.tensor([classes.index(p.parent.name) for p in te])
    configs = ["base", "reg_only"] + K_SWEEP
    config = {"task": "claim_e_test_time_registers", "det": DET, "k_sweep": K_SWEEP, "n_calib": N_CALIB,
              "n_train": N_TRAIN, "n_test": N_TEST, "seed": SEED, "device": device, "primary_k": PRIMARY_K,
              "torch": torch.__version__, "timm": timm.__version__, "commit": os.environ.get("ORX_COMMIT"),
              "test_sha1": hashlib.sha1("\n".join(p.name for p in te).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    results, failed = {}, {}
    for key in DET:
        t0 = time.time()
        try:
            h = Harness(key, device)
            h.calibrate(calib)
            layers_hist = {}
            for l, _, _ in h.ranked:
                layers_hist[l] = layers_hist.get(l, 0) + 1
            print(f"\n== {key}: calib outlier tokens={h.act_n}, top-{max(K_SWEEP)} register neurons by layer "
                  f"{dict(sorted(layers_hist.items()))}; top5={[(l, n, round(s, 2)) for l, n, s in h.ranked[:5]]}",
                  flush=True)
            res = {"calib_outlier_tokens": h.act_n, "ranked": h.ranked, "configs": {}}
            for c in configs:
                h.set_config(c)
                ftr, fte = h.features(tr), h.features(te, record=True)
                r = {"count": fte["count"], "max_ratio": fte["max_ratio"], "reg_ratio": fte["reg_ratio"],
                     "acc_mean": linear_probe(ftr["mean"], ytr, fte["mean"], yte)}
                if fte["cls"] is not None:
                    r["acc_cls"] = linear_probe(ftr["cls"], ytr, fte["cls"], yte)
                res["configs"][str(c)] = r
                last = h.L - 1
                print(f"  {str(c):>8}: outliers/img peak={max(r['count']):.2f} last={r['count'][last]:.2f} "
                      f"maxratio_last={r['max_ratio'][last]:.2f} reg_ratio={r['reg_ratio']} "
                      f"acc_mean={r['acc_mean']:.3f} acc_cls={r.get('acc_cls')}", flush=True)
                print(f"           count/layer={[round(v, 2) for v in r['count']]}", flush=True)
            results[key] = res
            del h
            torch.cuda.empty_cache() if device == "cuda" else None
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
