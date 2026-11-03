"""FIG: per-image sink maps on real photos, for the README.

N_IMG Imagenette val photos; each is center-cropped to a square once, and that same crop is
resized to every model's input size (so overlays line up across models). Per patch: L2 norm /
median patch norm at one block. Printed per (image, model, variant) as one RESULT_JSON line,
plus each square crop as a base64 JPEG split over IMG_CHUNK lines (log lines cap at 64 KiB).
"""

import base64
import gc
import io
import json
import os

import numpy as np
import timm
import torch
from PIL import Image

from measure_outliers import image_paths

N_IMG, SEED, SHOW = 4, 3, 256
# key: (timm name, block, sink (r, c) for the swap variant or None)
MODELS = {
    "clip_l14": ("vit_large_patch14_clip_224.openai", 16, None),
    "dinov2_l14": ("vit_large_patch14_dinov2.lvd142m", 20, None),
    "dinov3_l16": ("vit_large_patch16_dinov3.lvd1689m", 20, None),
    "aimv2_l14": ("aimv2_large_patch14_224.apple_pt", 12, (0, 0)),
    "siglip2_so400m16": ("vit_so400m_patch16_siglip_256.v2_webli", 20, (11, 0)),
}


def square(img):
    w, h = img.size
    s = min(w, h)
    return img.convert("RGB").crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def b64(img):
    buf = io.BytesIO()
    img.resize((SHOW, SHOW), Image.BICUBIC).save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def maps(model, crops, block, device):
    cfg = timm.data.resolve_data_config({}, model=model)
    S = cfg["input_size"][1]
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    x = torch.stack([(torch.from_numpy(np.array(c.resize((S, S), Image.BICUBIC))).permute(2, 0, 1).float() / 255
                      - mean) / std for c in crops]).to(device)
    out = {}
    h = model.blocks[block].register_forward_hook(lambda m, i, o: out.setdefault("o", o.float()))
    model.forward_features(x)
    h.remove()
    o, npt = out["o"], model.num_prefix_tokens
    n = o.norm(dim=-1)
    med = n[:, npt:].median(1).values[:, None]
    g = model.patch_embed.grid_size[0]
    return (n[:, npt:] / med).view(len(crops), g, g).cpu(), (n[:, :npt] / med).cpu()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = image_paths(N_IMG, SEED)  # val split on this branch
    crops = [square(Image.open(p)) for p in paths]
    print("config:", json.dumps({"task": "fig_sink_maps", "images": [p.name for p in paths], "models": MODELS,
                                 "commit": os.environ.get("ORX_COMMIT")}), flush=True)
    for i, c in enumerate(crops):
        s = b64(c)
        chunks = [s[j:j + 60000] for j in range(0, len(s), 60000)]
        for k, ch in enumerate(chunks):
            print("IMG_CHUNK " + json.dumps({"img": i, "part": k, "of": len(chunks), "data": ch}), flush=True)
    failed = {}
    for key, (name, block, sink) in MODELS.items():
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)
            variants = ["clean"] + (["swap"] if sink else [])
            for v in variants:
                if v == "swap":
                    g = model.patch_embed.grid_size[0]
                    pe = model.pos_embed.data
                    a, b = sink[0] * g + sink[1], 8 * g + 8
                    pe[0, [a, b]] = pe[0, [b, a]].clone()
                m, pre = maps(model, crops, block, device)
                for i in range(len(crops)):
                    top = int(m[i].argmax())
                    g = m.shape[-1]
                    line = {"img": i, "model": key, "variant": v, "block": block, "grid": g,
                            "top": [top // g, top % g], "top_ratio": round(float(m[i].max()), 2),
                            "prefix_ratio": [round(float(t), 2) for t in pre[i]],
                            "map": [[round(float(t), 3) for t in row] for row in m[i]]}
                    print("RESULT_JSON " + json.dumps(line), flush=True)
                tops = [(int(mm.argmax()) // g, int(mm.argmax()) % g) for mm in m]
                print(f"{key} {v}: top patches {tops} max ratio {[round(float(mm.max()), 1) for mm in m]}", flush=True)
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # keep the other models
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"!! {key} failed: {failed[key]}", flush=True)
    print("failed:", json.dumps(failed))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
