"""Claim D-VLM: does the constant-register fix (run D) work inside LLaVA-OneVision?

Tower sink = patch (21,2) (runs V, V3). Register = mean clean layer-0 input at the sink over
N_CAL Imagenette train images (disjoint from the eval sets), appended to the tower's token
sequence before layer 0 and dropped before the projector. All 729 patch tokens go to the LLM.
Variants: full | kill (mean pos embed at the sink) | kill+reg | clean+reg | kill+zero.
Same 1100 POPE/GQA items as V/V2; paired bootstrap 95% CIs vs full.
"""

import hashlib
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from measure_outliers import image_paths
from vlm_pruning import VLM, boot, correct, load_items

SINK = 21 * 27 + 2
N_CAL, SMOKE = 100, 4
VARIANTS = ["full", "kill", "kill+reg", "clean+reg", "kill+zero"]


class RegVLM(VLM):
    def __init__(self, device):
        super().__init__(device)
        enc = self.vt.encoder if hasattr(self.vt, "encoder") else self.vt.vision_model.encoder
        enc.layers[0].register_forward_pre_hook(self._pre)
        self.reg, self.record = None, None
        self.N = self.g * self.g

    def _pre(self, m, args):
        x = args[0]
        if self.record is not None:
            self.record.append(x[:, SINK].float().sum(0))
        if self.reg is None:
            return None
        return (torch.cat([x, self.reg.to(x.dtype).expand(x.shape[0], 1, -1)], 1),) + args[1:]

    def _vision(self, img):
        feats, ratio, dom = super()._vision(img)
        return feats[:self.N], ratio[:, :self.N], dom[:self.N]  # drop the appended register

    @torch.no_grad()
    def calibrate(self, paths):
        self.reg, self.record = None, []
        for p in paths:
            self.vision(Image.open(p))
        self.reg_mean = (torch.stack(self.record).sum(0) / len(paths)).to(self.device)
        self.record = None

    def encode(self, img, variant):
        kill = SINK if variant.startswith("kill") else None
        self.reg = {"reg": self.reg_mean, "zero": torch.zeros_like(self.reg_mean)}.get(variant.partition("+")[2])
        try:
            return self.vision(img, kill=kill)
        finally:
            self.reg = None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    vlm = RegVLM(device)
    cal = image_paths(N_CAL, 0, "train")
    vlm.calibrate(cal)
    items = load_items()
    print("config:", json.dumps({"task": "claim_d_vlm", "sink": SINK, "n_cal": N_CAL, "variants": VARIANTS,
                                 "n_items": len(items), "device": device, "commit": os.environ.get("ORX_COMMIT"),
                                 "items_sha1": hashlib.sha1("\n".join(i[1] + i[3] for i in items).encode()).hexdigest()[:12]}),
          flush=True)
    rows, tasks, ratios = {v: [] for v in VARIANTS}, [], {v: [] for v in VARIANTS}
    idx = torch.arange(vlm.N, device=device)
    cache_key, enc = None, None
    for n, (task, key, img, q, gold) in enumerate(items):
        if key != cache_key:
            enc = {v: vlm.encode(img, v) for v in VARIANTS}
            cache_key = key
        tasks.append(task)
        for v in VARIANTS:
            f, r, _ = enc[v]
            ratios[v].append(float(r[-1][SINK]))
            pred = vlm.answer(f, idx, q)
            rows[v].append(int(correct(task, pred, gold)))
            if n < SMOKE:
                print(f"  smoke [{task}] {v:>10}: sink ratio last block={float(r[-1][SINK]):.2f} pred={pred!r} "
                      f"gold={gold!r}", flush=True)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(items)} " + " ".join(f"{v}={np.mean(rows[v]):.3f}" for v in VARIANTS) +
                  f" ({time.time() - t0:.0f}s)", flush=True)
    print("mean sink-patch norm ratio at the last block: " +
          json.dumps({v: round(float(np.mean(ratios[v])), 2) for v in VARIANTS}), flush=True)
    for task in ("pope", "gqa", "all"):
        m = [t == task or task == "all" for t in tasks]
        sub = {v: [x for x, keep in zip(rows[v], m) if keep] for v in VARIANTS}
        for v in VARIANTS:
            line = {"task": task, "variant": v, "n": len(sub[v]), "acc": float(np.mean(sub[v]))}
            if v != "full":
                line["vs_full"] = boot(sub[v], sub["full"])
            print("RESULT_JSON " + json.dumps(line), flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
