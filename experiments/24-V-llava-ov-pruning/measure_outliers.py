"""Claim A: per-layer outlier (high-norm) patch tokens in frozen ViT encoders.

For each encoder, run frozen forward passes on a fixed Imagenette val subset at the
model's native resolution, hook every transformer block, and record the L2 norm of
every residual-stream token. A patch token is an outlier when its norm exceeds
K x the median patch norm of the same image at the same layer.

Outputs (in outputs/): summary.json, norms_<key>.npz (patch norms, float16,
shape [images, layers, patches], plus prefix-token norms).
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# hf-xet downloads hung indefinitely on this machine (run 69cefb78); plain HTTP works.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import argparse
import hashlib
import json
import random
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import constants as hf_constants
import timm
import torch
from PIL import Image

MODELS = {  # Claim K3: SigLIP towers used in open VLMs (fixed sinks found in SigLIP2 So400m/g, run K)
    "siglip1_so400m14_224": "vit_so400m_patch14_siglip_224.webli",
    "siglip1_so400m14_384": "vit_so400m_patch14_siglip_384.webli",
    "siglip1_l16_256": "vit_large_patch16_siglip_256.webli",
    "siglip2_so400m14_224": "vit_so400m_patch14_siglip_224.v2_webli",
    "siglip2_so400m14_378": "vit_so400m_patch14_siglip_378.v2_webli",
    "siglip2_so400m16_384": "vit_so400m_patch16_siglip_384.v2_webli",
    "siglip2_so400m16_512": "vit_so400m_patch16_siglip_512.v2_webli",
    "paligemma1_224": "vit_so400m_patch14_siglip_gap_224.pali_pt",
    "paligemma2_3b_224": "vit_so400m_patch14_siglip_gap_224.pali2_3b_pt",
    "paligemma2_3b_448": "vit_so400m_patch14_siglip_gap_448.pali2_3b_pt",
}
THRESHOLDS = (3.0, 4.0, 6.0)
PRIMARY_K = 4.0
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/{}.tgz"
DATA_DIR = Path.home() / ".cache" / "vision-encoder-anatomy"


def image_paths(n: int, seed: int, split: str = "val", dataset: str = "imagenette2-320") -> list[Path]:
    root = DATA_DIR / dataset
    if not root.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tgz = DATA_DIR / f"{dataset}.tgz"
        url = DATA_URL.format(dataset)
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(DATA_DIR, filter="data")
        tgz.unlink()
    paths = sorted((root / split).rglob("*.JPEG"))
    if len(paths) < n:
        raise SystemExit(f"only {len(paths)} val images, asked for {n}")
    return sorted(random.Random(seed).sample(paths, n))


def outlier_stats(patch: np.ndarray) -> dict:
    """patch: [images, layers, patches] norms. Returns per-layer lists."""
    patch = patch.astype(np.float32)
    ratio = patch / np.median(patch, axis=2, keepdims=True)
    out = {
        "median_norm": np.median(patch, axis=(0, 2)).tolist(),
        "max_ratio_mean": ratio.max(axis=2).mean(axis=0).tolist(),
        "p99_ratio": np.percentile(ratio, 99, axis=(0, 2)).tolist(),
    }
    for k in THRESHOLDS:
        m = ratio > k
        out[f"frac_tokens_gt{k:g}x"] = m.mean(axis=(0, 2)).tolist()
        out[f"frac_images_gt{k:g}x"] = m.any(axis=2).mean(axis=0).tolist()
        out[f"mean_count_gt{k:g}x"] = m.sum(axis=2).mean(axis=0).tolist()
    # Where do primary-threshold outliers sit? One position in every image = positional artifact.
    pos_counts = (ratio > PRIMARY_K).sum(axis=0)  # [layers, patches]
    total = pos_counts.sum(axis=1)
    out["top_pos"] = pos_counts.argmax(axis=1).tolist()
    out["top_pos_share"] = [float(c.max() / t) if t else None for c, t in zip(pos_counts, total)]
    out["top_pos_frac_images"] = (pos_counts.max(axis=1) / patch.shape[0]).tolist()
    out["n_outlier_positions"] = (pos_counts > 0).sum(axis=1).tolist()
    return out


@torch.no_grad()
def measure(key: str, name: str, paths: list[Path], device: str, bs: int, out_dir: Path) -> dict:
    t0 = time.time()
    model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    npt = int(getattr(model, "num_prefix_tokens", 0))
    blocks = model.blocks
    grabbed: list[torch.Tensor] = []
    hooks = [b.register_forward_hook(lambda m, i, o: grabbed.append(o.float().norm(dim=-1).cpu()))
             for b in blocks]
    per_batch = []
    for s in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[s:s + bs]]).to(device)
        grabbed.clear()
        model.forward_features(x)
        if len(grabbed) != len(blocks):
            raise RuntimeError(f"{key}: got {len(grabbed)} hook outputs for {len(blocks)} blocks")
        per_batch.append(torch.stack(grabbed, dim=1))  # [b, layers, tokens]
    for h in hooks:
        h.remove()
    norms = torch.cat(per_batch).numpy()
    prefix, patch = norms[:, :, :npt], norms[:, :, npt:]
    grid = cfg["input_size"][1] // model.patch_embed.patch_size[0]
    if patch.shape[2] != grid * grid:
        raise RuntimeError(f"{key}: {patch.shape[2]} patch tokens, expected {grid}x{grid}")
    stats = outlier_stats(patch)
    patch_med = np.median(patch, axis=2)  # [images, layers]
    stats["prefix_ratio_mean"] = [(prefix[:, :, j] / patch_med).mean(axis=0).tolist() for j in range(npt)]
    info = {"timm_name": name, "input_size": list(cfg["input_size"]), "num_layers": len(blocks),
            "num_prefix_tokens": npt, "num_patches": int(patch.shape[2]),
            "seconds": round(time.time() - t0, 1), "stats": stats}
    del model
    # 18 checkpoints up to 3B params: drop each one's weights after use to bound disk
    shutil.rmtree(Path(hf_constants.HF_HUB_CACHE) / f"models--timm--{name}", ignore_errors=True)
    if device == "cuda":
        torch.cuda.empty_cache()
    if device == "mps":
        torch.mps.empty_cache()
    return info


def print_summary(key: str, info: dict) -> None:
    s = info["stats"]
    tok = s[f"frac_tokens_gt{PRIMARY_K:g}x"]
    img = s[f"frac_images_gt{PRIMARY_K:g}x"]
    peak = int(np.argmax(tok))
    first = next((i for i, v in enumerate(img) if v >= 0.5), None)
    print(f"\n== {key} ({info['timm_name']}) input={info['input_size']} layers={info['num_layers']} "
          f"prefix={info['num_prefix_tokens']} patches={info['num_patches']} {info['seconds']}s")
    print(f"  peak layer {peak}: frac_tokens>{PRIMARY_K:g}x={tok[peak]:.4f} "
          f"frac_images={img[peak]:.3f} max_ratio_mean={s['max_ratio_mean'][peak]:.2f}")
    print(f"  first layer with >=50% images having an outlier: {first}")
    g = int(round(info["num_patches"] ** 0.5))
    print("  layer: frac_tokens>4x | frac_images>4x | max_ratio_mean | top_pos(r,c) | top_pos_share | top_pos_frac_images | n_positions")
    for i in range(info["num_layers"]):
        tp, sh = s["top_pos"][i], s["top_pos_share"][i]
        print(f"  {i:2d}: {tok[i]:.4f} | {img[i]:.3f} | {s['max_ratio_mean'][i]:.2f} | "
              f"({tp // g},{tp % g}) | {'-' if sh is None else f'{sh:.3f}'} | "
              f"{s['top_pos_frac_images'][i]:.3f} | {s['n_outlier_positions'][i]}")
    if info["num_prefix_tokens"]:
        last = [round(r[-1], 2) for r in s["prefix_ratio_mean"]]
        print(f"  prefix-token norm / patch median at last layer: {last}")


def selftest() -> None:
    a = np.ones((2, 1, 10), np.float32)
    a[0, 0, 3] = 10.0  # one 10x outlier in image 0
    s = outlier_stats(a)
    assert s["frac_tokens_gt4x"] == [0.05] and s["frac_images_gt4x"] == [0.5]
    assert s["mean_count_gt6x"] == [0.5] and s["max_ratio_mean"] == [5.5]
    assert s["top_pos"] == [3] and s["top_pos_share"] == [1.0] and s["n_outlier_positions"] == [1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    selftest()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    keys = args.models.split(",")
    unknown = set(keys) - set(MODELS)
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    paths = image_paths(args.n_images, args.seed)
    config = {**vars(args), "device": device, "torch": torch.__version__, "timm": timm.__version__,
              "thresholds": THRESHOLDS, "primary_k": PRIMARY_K, "commit": os.environ.get("ORX_COMMIT"),
              "images_sha1": hashlib.sha1("\n".join(p.name for p in paths).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    results, failed = {}, {}
    for key in keys:
        try:
            results[key] = measure(key, MODELS[key], paths, device, args.batch_size, out_dir)
            print_summary(key, results[key])
        except Exception as e:  # keep the other encoders' results if one fails to load
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"\n!! {key} failed: {failed[key]}", flush=True)
        (out_dir / "summary.json").write_text(json.dumps({"config": config, "results": results,
                                                           "failed": failed}, indent=1))
    print("\n=== SUMMARY_JSON_BEGIN ===")
    print(json.dumps({"config": config, "results": results, "failed": failed}))
    print("=== SUMMARY_JSON_END ===")
    print(f"\nwrote {out_dir.resolve()}/summary.json; ok={list(results)} failed={list(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    # Claim V node: the fixed run command now runs the LLaVA-OneVision pruning test.
    import vlm_pruning

    vlm_pruning.main()
