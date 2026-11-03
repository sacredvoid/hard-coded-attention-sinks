"""Claim E2: do test-time registers help dense prediction?

Register neurons are calibrated exactly as in E (Imagenette train images). Dense probe:
ADE20K, images and labels resized square to the encoder's native resolution, labels
majority-pooled to the patch grid (class 0 'other' ignored). Frozen final patch tokens
(register token dropped) -> one linear layer per patch. Same probe hyperparameters for every
encoder and config. Reports patch-grid mIoU / accuracy and accuracy at outlier vs normal
patch positions (last-block norm > K x median).
"""

import hashlib
import json
import os
import random
import time
import urllib.request
import zipfile

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from measure_outliers import DATA_DIR, PRIMARY_K, image_paths
from test_time_registers import DET, N_CALIB, SEED, Harness

ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
N_TR, N_TE, PATCHES_PER_TRAIN_IMG, N_CLS = 1000, 500, 256, 150
CONFIGS = ["base", 10, 20, 50]


def ade_pairs(split, n):
    root = DATA_DIR / "ADEChallengeData2016"
    if not root.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        z = DATA_DIR / "ade.zip"
        print(f"downloading {ADE_URL}", flush=True)
        urllib.request.urlretrieve(ADE_URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA_DIR)
        z.unlink()
    imgs = sorted((root / "images" / split).glob("*.jpg"))
    pairs = [(p, root / "annotations" / split / (p.stem + ".png")) for p in imgs]
    pairs = random.Random(SEED).sample(pairs, n)
    missing = [a for _, a in pairs if not a.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ADE annotations missing, e.g. {missing[0]}")
    return pairs


def patch_labels(lbl, g):
    """lbl: [S,S] long in 0..150. Majority class (0..149) per patch, -1 where all 'other'."""
    p = lbl.shape[0] // g
    oh = F.one_hot(lbl[:g * p, :g * p], N_CLS + 1)[..., 1:].float()  # drop class 0
    cnt = oh.view(g, p, g, p, N_CLS).sum((1, 3)).view(g * g, N_CLS)
    return torch.where(cnt.sum(1) > 0, cnt.argmax(1), torch.full_like(cnt[:, 0], -1, dtype=torch.long))


@torch.no_grad()
def extract(h, pairs, mean, std, S, train):
    g = S // h.model.patch_embed.patch_size[0]
    feats, labels, outl = [], [], []
    gen = torch.Generator().manual_seed(SEED)
    for s in range(0, len(pairs), 8):
        batch = pairs[s:s + 8]
        x = torch.stack([torch.from_numpy(
            np.array(Image.open(i).convert("RGB").resize((S, S), Image.BICUBIC))).permute(2, 0, 1)
            for i, _ in batch]).float().div(255).to(h.device)
        x = (x - mean) / std
        out = h.model.forward_features(x).float()
        if h.reg:
            out = out[:, :-1]
        pt = out[:, h.npt:]  # [b, g*g, D]
        ratio = h.ratio[-1]
        for j, (_, a) in enumerate(batch):
            lbl = torch.from_numpy(np.array(Image.open(a).resize((S, S), Image.NEAREST))).long()
            y = patch_labels(lbl, g)
            keep = (y >= 0).nonzero().squeeze(1)
            if train and len(keep) > PATCHES_PER_TRAIN_IMG:
                keep = keep[torch.randperm(len(keep), generator=gen)[:PATCHES_PER_TRAIN_IMG]]
            feats.append(pt[j, keep.to(pt.device)].cpu())
            labels.append(y[keep])
            outl.append((ratio[j, keep.to(ratio.device)] > PRIMARY_K).cpu())
    return torch.cat(feats), torch.cat(labels), torch.cat(outl)


def fit_eval(xtr, ytr, xte, yte, device):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = ((xtr - mu) / sd).to(device), ((xte - mu) / sd).to(device)
    ytr = ytr.to(device)
    lin = torch.nn.Linear(xtr.shape[1], N_CLS).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(15):
        perm = torch.randperm(len(xtr), generator=g).to(device)
        for s in range(0, len(xtr), 4096):
            i = perm[s:s + 4096]
            loss = F.cross_entropy(lin(xtr[i]), ytr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = torch.cat([lin(xte[s:s + 65536]).argmax(1) for s in range(0, len(xte), 65536)]).cpu()
    return pred


def miou(pred, y):
    cm = torch.bincount(y * N_CLS + pred, minlength=N_CLS * N_CLS).view(N_CLS, N_CLS).double()
    inter, union = cm.diag(), cm.sum(0) + cm.sum(1) - cm.diag()
    present = cm.sum(1) > 0
    return (inter[present] / union[present]).mean().item()


def selftest():
    lbl = torch.zeros(4, 4, dtype=torch.long)
    lbl[:2, :2] = 3
    lbl[0, 0] = 5  # patch (0,0): three 3s, one 5 -> class index 2
    y = patch_labels(lbl, 2)
    assert y.tolist() == [2, -1, -1, -1], y
    p, t = torch.tensor([0, 1, 1]), torch.tensor([0, 1, 0])
    assert abs(miou(p, t) - 0.5) < 1e-9  # class0 IoU 1/2, class1 IoU 1/2


def main():
    selftest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calib = image_paths(N_CALIB + 1000, SEED, "train")
    random.Random(SEED).shuffle(calib)  # same calibration images as E
    calib = calib[:N_CALIB]
    tr, te = ade_pairs("training", N_TR), ade_pairs("validation", N_TE)
    config = {"task": "claim_e2_dense_probe", "n_tr": N_TR, "n_te": N_TE, "patches_per_train_img": PATCHES_PER_TRAIN_IMG,
              "configs": CONFIGS, "det": DET, "primary_k": PRIMARY_K, "device": device, "torch": torch.__version__,
              "timm": timm.__version__, "commit": os.environ.get("ORX_COMMIT"),
              "te_sha1": hashlib.sha1("\n".join(p.name for p, _ in te).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    results, failed = {}, {}
    for key in DET:
        t0 = time.time()
        try:
            h = Harness(key, device)
            h.calibrate(calib)
            cfg = timm.data.resolve_data_config({}, model=h.model)
            S = cfg["input_size"][1]
            mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
            std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
            print(f"\n== {key} S={S}", flush=True)
            res = {}
            for c in CONFIGS:
                h.set_config(c)
                xtr, ytr, _ = extract(h, tr, mean, std, S, True)
                xte, yte, ote = extract(h, te, mean, std, S, False)
                pred = fit_eval(xtr, ytr, xte, yte, device)
                ok = pred == yte
                r = {"miou": miou(pred, yte), "acc": ok.float().mean().item(),
                     "n_test_patches": len(yte), "outlier_patches": int(ote.sum()),
                     "acc_at_outliers": ok[ote].float().mean().item() if ote.any() else None,
                     "acc_at_normal": ok[~ote].float().mean().item()}
                res[str(c)] = r
                print(f"  {str(c):>5}: mIoU={r['miou']:.4f} acc={r['acc']:.4f} outlier_patches={r['outlier_patches']} "
                      f"acc@outlier={r['acc_at_outliers']} acc@normal={r['acc_at_normal']:.4f}", flush=True)
            results[key] = res
            del h
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
