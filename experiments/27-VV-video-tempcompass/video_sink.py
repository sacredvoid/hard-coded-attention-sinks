"""Claim VV: does dropping the per-frame sink help video QA? (SToP's setting)

LLaVA-OneVision-0.5B on TempCompass multi-choice (1580 questions, 410 videos), 16 uniform frames.
Each frame goes through the SigLIP tower as in HF: block -1 features -> projector -> bilinear
27x27 -> 14x14 pooling (196 tokens/frame); one image_newline after the last frame. Encoder
attention received at block -2 is pooled the same way to score pooled tokens. The pooled sink
index is the argmax of mean pooled attention over the first frames seen.
Variants: full | full_minus_sink (pooled sink dropped in every frame) | kill_full (sink killed
inside the tower via mean pos embed at (21,2)) | per-frame topk@k, topk_nosink@k, random@k for
k = 8, 16. Letter-match accuracy; paired bootstrap 95% CIs.
"""

import hashlib
import json
import os
import random
import re
import time
import zipfile
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from measure_outliers import DATA_DIR
from vlm_pruning import VLM, boot

T, KS, SEED, N_CAL_VIDEOS, SMOKE = 16, (8, 16), 0, 30, 6
SUFFIX = "\nAnswer with the option's letter from the given choices directly."
SINK_FULL = 21 * 27 + 2  # LLaVA-OV tower sink (runs V, V3)


def video_dir():
    root = DATA_DIR / "tempcompass"
    if not root.exists():
        z = hf_hub_download("lmms-lab/TempCompass", "tempcompass_videos.zip", repo_type="dataset")
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
    return root


def frames(path, n=T):
    with av.open(str(path)) as c:
        fr = [f.to_image() for f in c.decode(video=0)]
    if not fr:
        raise RuntimeError(f"no frames decoded from {path}")
    idx = np.linspace(0, len(fr) - 1, n).round().astype(int)
    return [fr[i] for i in idx]


def pool_feats(x, g):
    """x: [frames, g*g, D] -> [frames, ceil(g/2)^2, D], as HF apply_pooling."""
    n, _, d = x.shape
    x = x.view(n, g, g, d).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(-(-g // 2), -(-g // 2)), mode="bilinear")
    return x.permute(0, 2, 3, 1).reshape(n, -1, d)


def letter(s):
    s = s.strip().upper()
    m = re.match(r"\(?([A-D])(?:[.):,]|$)", s) or re.search(r"ANSWER IS \(?([A-D])(?:[.):,\s]|$)", s)
    return m.group(1) if m else ""


class VideoVLM(VLM):
    @torch.no_grad()
    def encode(self, imgs, kill=None):
        f, d = [], []
        for im in imgs:
            feats, _, dom = self.vision(im, kill=kill)
            f.append(feats)
            d.append(dom)
        f = pool_feats(self.proj(torch.stack(f)), self.g)  # [T, 196, D_text]
        d = pool_feats(torch.stack(d)[..., None].float(), self.g)[..., 0]  # [T, 196]
        return f, d

    @torch.no_grad()
    def answer_video(self, f, idx_per_frame, question, max_new=4):
        core = self.model.model if hasattr(self.model, "model") else self.model
        toks = torch.cat([f[t, idx.to(f.device)] for t, idx in enumerate(idx_per_frame)])
        toks = torch.cat([toks, core.image_newline[None].to(toks.dtype)])
        conv = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True)
        pre, post = text.split(self.proc.video_token)
        ids = [self.tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device) for t in (pre, post)]
        emb = torch.cat([self.embed(ids[0]), toks[None].to(self.embed.weight.dtype), self.embed(ids[1])], 1)
        out = self.model.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=self.device,
                                  dtype=torch.long), max_new_tokens=max_new, do_sample=False)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


def select_frame(variant, k, dom, sink, rng):
    n = len(dom)
    if variant == "full":
        return torch.arange(n)
    if variant == "full_minus_sink":
        return torch.tensor([i for i in range(n) if i != sink])
    if variant == "random":
        return torch.tensor(sorted(rng.sample(range(n), k)))
    s = dom.clone()
    if variant == "topk_nosink":
        s[sink] = -1
    return s.topk(k).indices.sort().values.cpu()


def selftest():
    x = torch.arange(2 * 27 * 27 * 3, dtype=torch.float).view(2, 27 * 27, 3)
    assert pool_feats(x, 27).shape == (2, 196, 3)
    assert letter("B. walking") == "B" and letter("The answer is C") == "C" and letter("a dog") == ""
    d = torch.tensor([0.1, 0.9, 0.5, 0.8])
    assert select_frame("topk_nosink", 2, d, 1, random.Random(0)).tolist() == [2, 3]
    assert select_frame("full_minus_sink", 0, d, 1, random.Random(0)).tolist() == [0, 2, 3]


def main():
    selftest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    qs = load_dataset("lmms-lab/TempCompass", "multi-choice", split="test")
    root = video_dir()
    files = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in (".mp4", ".webm", ".mkv", ".avi", ".mov")}
    items = [(r["video_id"], r["question"] + SUFFIX, r["answer"].strip()[0], r["dim"]) for r in qs]
    missing = sorted({v for v, *_ in items if v not in files})
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos missing, e.g. {missing[:3]}")
    items.sort(key=lambda r: r[0])  # group by video so each video is encoded once
    vlm = VideoVLM(device)
    variants = ["full", "full_minus_sink", "kill_full"] + \
        [f"{v}@{k}" for k in KS for v in ("topk", "topk_nosink", "random")]
    print("config:", json.dumps({"task": "claim_vv_video_sink", "n_q": len(items), "n_videos": len(files), "frames": T,
                                 "ks": KS, "variants": variants, "device": device, "commit": os.environ.get("ORX_COMMIT"),
                                 "items_sha1": hashlib.sha1("\n".join(i[0] + i[1] for i in items).encode()).hexdigest()[:12]}),
          flush=True)
    # pooled sink index from mean pooled attention over the first videos
    acc_dom, vids = None, sorted({v for v, *_ in items})[:N_CAL_VIDEOS]
    for v in vids:
        _, d = vlm.encode(frames(files[v]))
        acc_dom = d.sum(0) if acc_dom is None else acc_dom + d.sum(0)
    sink = int(acc_dom.argmax())
    print(f"pooled sink index = {sink} = ({sink // 14},{sink % 14}) [full-grid sink (21,2) -> expected near (11,1)=155]",
          flush=True)
    rows = {v: [] for v in variants}
    dims, rng, cur, enc = [], random.Random(SEED), None, None
    t1 = time.time()
    for n_done, (vid, q, gold, dim) in enumerate(items):
        if vid != cur:
            imgs = frames(files[vid])
            enc = vlm.encode(imgs), vlm.encode(imgs, kill=SINK_FULL)
            cur = vid
        (f, d), (fk, _) = enc
        dims.append(dim)
        for v in variants:
            name, k = (v.split("@") + [None])[:2]
            feats = fk if name == "kill_full" else f
            name = "full" if name == "kill_full" else name
            idx = [select_frame(name, int(k) if k else 0, d[t], sink, rng) for t in range(T)]
            pred = vlm.answer_video(feats, idx, q)
            rows[v].append(int(letter(pred) == gold))
            if n_done < SMOKE:
                print(f"  smoke {v:>16}: tokens={sum(len(i) for i in idx)} pred={pred!r} gold={gold}", flush=True)
        if (n_done + 1) % 200 == 0:
            el = time.time() - t1
            print(f"  {n_done + 1}/{len(items)} done, {el / (n_done + 1):.2f}s/q; full={np.mean(rows['full']):.3f} "
                  f"minus_sink={np.mean(rows['full_minus_sink']):.3f} kill={np.mean(rows['kill_full']):.3f} "
                  f"topk@8={np.mean(rows['topk@8']):.3f} nosink@8={np.mean(rows['topk_nosink@8']):.3f} "
                  f"random@8={np.mean(rows['random@8']):.3f}", flush=True)
    for dim in sorted(set(dims)) + ["all"]:
        m = [x == dim or dim == "all" for x in dims]
        sub = {v: [r for r, keep in zip(rows[v], m) if keep] for v in variants}
        for v in variants:
            line = {"dim": dim, "variant": v, "n": len(sub[v]), "acc": float(np.mean(sub[v]))}
            if v != "full":
                line["vs_full"] = boot(sub[v], sub["full"])
            if "@" in v and not v.startswith("topk@"):
                line["vs_topk"] = boot(sub[v], sub["topk@" + v.split("@")[1]])
            print("RESULT_JSON " + json.dumps(line), flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
