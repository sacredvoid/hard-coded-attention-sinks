"""Claim F: are outlier (high-norm) patch tokens brittle under tiny input changes?

Per encoder, on N val images, compare outlier positions (norm > K x median patch norm, per
layer) between the clean image and perturbed versions: Gaussian pixel noise, JPEG q75,
half-patch translation, horizontal flip (positions mirrored before matching).
Match metrics at every layer:
  kept  = share of clean outliers with a perturbed outlier within 1 patch (Chebyshev)
  new   = share of perturbed outliers with no clean outlier within 1 patch
  count = mean outliers per image
Then PGD at the attack layer (L_ATK), eps in EPS, 10 steps:
  create: raise norm/median of a random normal patch (clean ratio < 2); success = ratio > K
  remove: push every patch ratio under 2; success = no patch > K at L_ATK
"""

import hashlib
import io
import json
import os
import time

import timm
import torch
import torch.nn.functional as F
from PIL import Image

from measure_outliers import MODELS, PRIMARY_K, image_paths

N_IMG, N_ATK, BS, SEED = 300, 100, 16, 0
NOISE = (1, 4, 16)  # sigma in /255 pixel units
EPS = (1, 4)  # L-inf budget in /255
PGD_STEPS = 10
# Attack layer: outlier plateau from the positions run (12 for encoders without patch outliers).
L_ATK = {"clip_l14": 13, "dinov2_l14": 19, "dinov2_l14_reg4": 19, "siglip2_l16": 15,
         "dinov3_l16": 12, "pe_core_l14": 12, "aimv2_l14": 20}


class Stop(Exception):
    pass


class Enc:
    def __init__(self, key, device):
        self.device = device
        self.model = timm.create_model(MODELS[key], pretrained=True, num_classes=0).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = timm.data.create_transform(**cfg)
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.npt = int(getattr(self.model, "num_prefix_tokens", 0))
        self.L = len(self.model.blocks)
        self.g = cfg["input_size"][1] // self.model.patch_embed.patch_size[0]
        self.stop_at = None
        self.out: dict[int, torch.Tensor] = {}
        for l, b in enumerate(self.model.blocks):
            b.register_forward_hook(self._hook(l))

    def _hook(self, l):
        def fn(m, i, o):
            n = o[:, self.npt:].float().norm(dim=-1)
            self.out[l] = n / n.median(dim=1, keepdim=True).values
            if self.stop_at == l:
                raise Stop
        return fn

    def pixels(self, imgs):  # PIL list -> [b,3,H,W] in [0,1]
        x = torch.stack([self.tf(im) for im in imgs]).to(self.device)
        return x * self.std + self.mean

    def ratios(self, pix, upto=None):
        """pix in [0,1]. Ascent-only callers negate their loss. Returns [b, L', patches] ratio tensor (layers 0..upto)."""
        self.stop_at, self.out = upto, {}
        try:
            self.model.forward_features((pix - self.mean) / self.std)
        except Stop:
            pass
        finally:
            self.stop_at = None
        return torch.stack([self.out[l] for l in sorted(self.out)], 1)


def neighbor_hits(a, b, g):
    """a,b: [b, P] bool on a g x g grid. Share of a's True cells with a True in b within 1 cell."""
    bb = F.max_pool2d(b.float().view(-1, 1, g, g), 3, 1, 1).view(b.shape) > 0
    return (a & bb).sum().item(), a.sum().item()


def jpeg(im, q=75):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


@torch.no_grad()
def stability(enc, paths, gen):
    g, K = enc.g, PRIMARY_K
    pert = [f"noise{s}" for s in NOISE] + ["jpeg75", "flip", "shift_half_patch"]
    acc = {k: [{"kept": [0, 0], "new": [0, 0], "count": 0.0} for _ in range(enc.L)] for k in ["clean"] + pert}
    half = enc.model.patch_embed.patch_size[0] // 2
    for s in range(0, len(paths), BS):
        ims = [Image.open(p).convert("RGB") for p in paths[s:s + BS]]
        x = enc.pixels(ims)
        rc = enc.ratios(x) > K  # [b, L, P]
        for l in range(enc.L):
            acc["clean"][l]["count"] += rc[:, l].sum().item()
        for name in pert:
            if name.startswith("noise"):
                sig = int(name[5:]) / 255
                xp = (x + sig * torch.randn(x.shape, generator=gen, device=x.device)).clamp(0, 1)
            elif name == "jpeg75":
                xp = enc.pixels([jpeg(im) for im in ims])
            elif name == "flip":
                xp = x.flip(-1)
            else:  # shift content right by half a patch in model-input pixels
                xp = torch.cat([enc.mean.expand(len(x), 3, x.shape[2], half), x[..., :-half]], -1)
            rp = enc.ratios(xp) > K
            if name == "flip":  # mirror positions back to the clean frame
                rp = rp.view(*rp.shape[:2], g, g).flip(-1).view(rp.shape)
            for l in range(enc.L):
                a = acc[name][l]
                h, n = neighbor_hits(rc[:, l], rp[:, l], g)
                a["kept"][0] += h
                a["kept"][1] += n
                h, n = neighbor_hits(rp[:, l], rc[:, l], g)
                a["new"][0] += n - h
                a["new"][1] += n
                a["count"] += rp[:, l].sum().item()
    out = {}
    for name, per_layer in acc.items():
        out[name] = {"count": [a["count"] / len(paths) for a in per_layer]}
        if name != "clean":
            out[name]["kept"] = [a["kept"][0] / a["kept"][1] if a["kept"][1] else None for a in per_layer]
            out[name]["new"] = [a["new"][0] / a["new"][1] if a["new"][1] else None for a in per_layer]
    return out


def pgd(enc, x, loss_fn, eps, la):
    alpha = eps / 4
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(PGD_STEPS):
        r = enc.ratios((x + delta).clamp(0, 1), upto=la)[:, -1]
        loss = loss_fn(r)
        (g,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta += alpha * g.sign()
            delta.clamp_(-eps, eps)
    with torch.no_grad():
        return enc.ratios((x + delta).clamp(0, 1), upto=la)[:, -1]


def attacks(enc, key, paths, gen):
    la = L_ATK[key]
    tokens = enc.g ** 2
    bs = max(1, min(8, 2048 // tokens))
    res = {}
    for e in EPS:
        eps = e / 255
        c_ok = c_n = r_ok = r_n = 0
        c_ratio = []
        for s in range(0, len(paths), bs):
            x = enc.pixels([Image.open(p).convert("RGB") for p in paths[s:s + bs]])
            with torch.no_grad():
                r0 = enc.ratios(x, upto=la)[:, -1]
            # create: random normal-token target per image
            w = (r0 < 2).float() + 1e-9
            tgt = torch.multinomial(w, 1, generator=gen).squeeze(1)
            ar = torch.arange(len(x), device=x.device)
            r = pgd(enc, x, lambda r: r[ar, tgt].sum(), eps, la)
            c_ok += (r[ar, tgt] > PRIMARY_K).sum().item()
            c_n += len(x)
            c_ratio += r[ar, tgt].tolist()
            # remove: only images that have an outlier at la
            has = (r0 > PRIMARY_K).any(1)
            if has.any():
                r = pgd(enc, x[has], lambda r: -F.relu(r - 2).sum(), eps, la)  # pgd ascends: negate
                r_ok += (~(r > PRIMARY_K).any(1)).sum().item()
                r_n += int(has.sum())
        res[f"eps{e}"] = {"create_success": c_ok / c_n, "create_median_ratio": float(torch.tensor(c_ratio).median()),
                          "remove_success": r_ok / r_n if r_n else None, "remove_n": r_n}
    return res


def selftest():
    a = torch.zeros(1, 9, dtype=torch.bool)
    b = torch.zeros(1, 9, dtype=torch.bool)
    a[0, 0] = True
    b[0, 4] = True  # (1,1) is within 1 cell of (0,0)
    assert neighbor_hits(a, b, 3) == (1, 1)
    b[0, 4], b[0, 8] = False, True  # (2,2) is not
    assert neighbor_hits(a, b, 3) == (0, 1)


def main():
    selftest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = image_paths(N_IMG, SEED)  # val split
    config = {"task": "claim_f_brittleness", "n_img": N_IMG, "n_atk": N_ATK, "noise": NOISE, "eps": EPS,
              "pgd_steps": PGD_STEPS, "l_atk": L_ATK, "primary_k": PRIMARY_K, "device": device,
              "torch": torch.__version__, "timm": timm.__version__, "commit": os.environ.get("ORX_COMMIT"),
              "images_sha1": hashlib.sha1("\n".join(p.name for p in paths).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    results, failed = {}, {}
    for key in MODELS:
        t0 = time.time()
        try:
            gen = torch.Generator(device=device).manual_seed(SEED)
            enc = Enc(key, device)
            st = stability(enc, paths, gen)
            la = L_ATK[key]
            print(f"\n== {key} (grid {enc.g}) stability at L{la} / last:", flush=True)
            for name, v in st.items():
                f = lambda arr, l: "-" if arr[l] is None else f"{arr[l]:.3f}"
                line = f"  {name:>16}: count={v['count'][la]:.2f}/{v['count'][-1]:.2f}"
                if "kept" in v:
                    line += f" kept={f(v['kept'], la)}/{f(v['kept'], -1)} new={f(v['new'], la)}/{f(v['new'], -1)}"
                print(line, flush=True)
            atk = attacks(enc, key, paths[:N_ATK], gen)
            for e, v in atk.items():
                print(f"  attack {e}: {json.dumps(v)}", flush=True)
            results[key] = {"stability": st, "attacks": atk}
            del enc
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
