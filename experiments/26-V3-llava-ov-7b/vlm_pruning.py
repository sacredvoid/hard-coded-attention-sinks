"""Claim V: the hard-coded SigLIP So400m sink inside LLaVA-OneVision, and a one-index pruning fix.

Model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf, single 384x384 view (no anyres crops, no
image_newline), fp16. Its SigLIP tower was fine-tuned during VLM training.

Part 1 (tower): per block, the most frequent top-norm patch and its share of images; the fixed
sink index is taken from the first N_CAL images. Where does the sink rank under VisionZip-style
'dominant' scores (attention received at block -2, mean over heads and queries)?
Part 2 (pruning): keep k of 729 visual tokens (raster order kept), project, and answer:
  topk         top-k by dominant score
  topk_nosink  same, never keeping the fixed sink index
  topk_nohinorm  same, never keeping tokens with norm > 4x median at the feature layer
                 (SinkPruner-style sanitizer; our threshold, not theirs)
  random       k uniform random tokens (seeded)
  full         all 729 tokens
POPE (yes/no) and GQA testdev-balanced (exact match). Paired bootstrap 95% CIs vs topk and full.
"""

import hashlib
import json
import os
import random
import re
import time

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"  # V3: replicate V2 on the 7B model
N_POPE, N_GQA, N_CAL, SMOKE, SEED = 600, 500, 100, 8, 0
KS = (16,)  # V3: only the k where V2 saw the large random-vs-topk gap
DOM_LAYER = -2  # VisionZip reads attention from the penultimate block
K_NORM = 4.0
GQA_SUFFIX = "\nAnswer the question using a single word or phrase."


class VLM:
    def __init__(self, device):
        self.device = device
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.float16).eval().to(device)
        core = self.model.model if hasattr(self.model, "model") else self.model
        self.vt, self.proj = core.vision_tower, core.multi_modal_projector
        self.vt.config._attn_implementation = "eager"  # attention weights needed; read at call time
        # transformers 5 SiglipVisionModel exposes encoder directly; older versions nest it in vision_model
        enc = self.vt.encoder if hasattr(self.vt, "encoder") else self.vt.vision_model.encoder
        layers = enc.layers
        self.attn = None
        layers[DOM_LAYER].self_attn.register_forward_hook(self._grab)
        self.embed = self.model.get_input_embeddings()
        self.tok = self.proc.tokenizer
        self.S = self.vt.config.image_size
        emb = self.vt.embeddings if hasattr(self.vt, "embeddings") else self.vt.vision_model.embeddings
        self.pe = emb.position_embedding.weight  # [729, D]
        self.g = self.S // self.vt.config.patch_size

    def _grab(self, m, i, o):
        if o[1] is None:
            raise RuntimeError("attention weights not returned; eager attention not active")
        self.attn = o[1]

    @torch.no_grad()
    def vision(self, img, kill=None):
        """kill = patch index whose position embedding is replaced by the mean one (V2), else None."""
        if kill is not None:
            saved = self.pe.data[kill].clone()
            self.pe.data[kill] = self.pe.data.mean(0)
        try:
            return self._vision(img)
        finally:
            if kill is not None:
                self.pe.data[kill] = saved

    def _vision(self, img):
        x = torch.from_numpy(np.array(img.convert("RGB").resize((self.S, self.S), Image.BICUBIC)))
        x = (x.permute(2, 0, 1).float() / 255 - 0.5) / 0.5
        out = self.vt(x[None].to(self.device, torch.float16), output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[1:]  # block outputs
        norms = torch.stack([h[0].float().norm(dim=-1) for h in hs])  # [L, 729]
        ratio = norms / norms.median(dim=1, keepdim=True).values
        dom = self.attn[0].float().mean(0).mean(0)  # [729] attention received
        return hs[-1][0], ratio, dom  # feature layer = -1 (config.vision_feature_layer)

    @torch.no_grad()
    def answer(self, feats, idx, question, max_new=10):
        img = self.proj(feats[idx][None])  # [1, k, D]
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True)
        pre, post = text.split(self.proc.image_token)
        ids = [self.tok(t, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device) for t in (pre, post)]
        emb = torch.cat([self.embed(ids[0]), img.to(self.embed.weight.dtype), self.embed(ids[1])], 1)
        out = self.model.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=self.device,
                                  dtype=torch.long), max_new_tokens=max_new, do_sample=False)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


def norm_ans(s):
    s = s.lower().strip().split("\n")[0]
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def correct(task, pred, gold):
    p, g = norm_ans(pred), norm_ans(gold)
    if task == "pope":
        return (p.startswith("yes")) == (g == "yes")
    return p == g


def select(variant, k, dom, ratio_feat, sink, rng):
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
    elif variant == "topk_nohinorm":
        s[ratio_feat > K_NORM] = -1
    return s.topk(k).indices.sort().values.cpu()


def boot(a, b, n=2000, seed=0):
    """Paired bootstrap of mean(a) - mean(b) over questions: (diff, lo, hi)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), (n, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return float((a - b).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load_items():
    pope = load_dataset("lmms-lab/POPE", split="test")
    pi = random.Random(SEED).sample(range(len(pope)), N_POPE)
    items = [("pope", f"pope{pope[i]['question_id']}", pope[i]["image"], pope[i]["question"], pope[i]["answer"])
             for i in pi]
    q = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
    ims = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    by_id = {r["id"]: r["image"] for r in ims}
    gi = random.Random(SEED).sample(range(len(q)), N_GQA)
    for i in gi:
        r = q[i]
        items.append(("gqa", f"gqa{r['imageId']}", by_id[r["imageId"]], r["question"] + GQA_SUFFIX, r["answer"]))
    random.Random(SEED).shuffle(items)  # interleave tasks so the smoke stage sees both
    return items


def selftest():
    assert correct("pope", "Yes, there is.", "yes") and not correct("pope", "No", "yes")
    assert correct("gqa", "The table.", "table") and not correct("gqa", "chair", "table")
    d = torch.tensor([0.1, 0.9, 0.5, 0.8])
    assert select("topk", 2, d, torch.ones(4), 1, random.Random(0)).tolist() == [1, 3]
    assert select("topk_nosink", 2, d, torch.ones(4), 1, random.Random(0)).tolist() == [2, 3]
    r = torch.tensor([1.0, 1.0, 1.0, 9.0])
    assert select("full_minus_sink", 0, d, r, 1, random.Random(0)).tolist() == [0, 2, 3]
    diff, lo, hi = boot([1, 1, 0, 1], [0, 1, 0, 1])
    assert abs(diff - 0.25) < 1e-9 and lo <= diff <= hi


def main():
    selftest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    vlm = VLM(device)
    items = load_items()
    # kill_* variants use features from the tower with the sink's position embedding replaced by the mean
    variants = ["full", "full_minus_sink", "kill_full", "kill_minus_sink"] + \
        [f"{v}@{k}" for k in KS for v in ("topk", "random")]
    config = {"task": "claim_v3_7b_replication", "model": MODEL_ID, "n_pope": N_POPE, "n_gqa": N_GQA, "n_cal": N_CAL,
              "ks": KS, "dom_layer": DOM_LAYER, "k_norm": K_NORM, "grid": vlm.g, "device": device,
              "torch": torch.__version__, "commit": os.environ.get("ORX_COMMIT"),
              "items_sha1": hashlib.sha1("\n".join(i[1] + i[3] for i in items).encode()).hexdigest()[:12]}
    print("config:", json.dumps(config), flush=True)
    print(f"loaded model + {len(items)} items in {time.time() - t0:.0f}s", flush=True)

    # Part 1: find the fixed sink in the fine-tuned tower from the first N_CAL distinct images
    cache, seen = {}, []
    for task, key, img, _, _ in items:
        if key not in cache:
            f, r, d = vlm.vision(img)
            cache[key] = (f, r.cpu(), d.cpu())
            seen.append(key)
        if len(seen) >= N_CAL:
            break
    L = cache[seen[0]][1].shape[0]
    top = torch.stack([cache[k][1].argmax(1) for k in seen])  # [n, L]
    modes, shares = [], []
    for l in range(L):
        v, c = top[:, l].unique(return_counts=True)
        modes.append(int(v[c.argmax()]))
        shares.append(float(c.max()) / len(seen))
    g = vlm.g
    print("tower top-norm patch per block (mode (r,c), share of images):", flush=True)
    print("  " + " ".join(f"{l}:({m // g},{m % g}){s:.2f}" for l, (m, s) in enumerate(zip(modes, shares))), flush=True)
    best = max(range(L), key=lambda l: shares[l])  # fallback: the most position-locked block
    sink = modes[-1] if shares[-1] >= 0.5 else modes[best]
    print(f"fixed sink index = {sink} = ({sink // g},{sink % g}); timm SigLIP v1 So400m@384 had (21,2)", flush=True)

    # Part 2: answer every item under every variant
    rows = {v: [] for v in variants}
    task_of, stats = [], {"sink_rank": []}
    rng = random.Random(SEED)
    t1 = time.time()
    for n_done, (task, key, img, question, gold) in enumerate(items):
        if key not in cache:
            f, r, d = vlm.vision(img)
            cache[key] = (f, r.cpu(), d.cpu())
        f, r, d = cache[key]
        rf = r[-1]
        fk, rk, _ = vlm.vision(img, kill=sink)
        if n_done < SMOKE:
            print(f"  kill check: sink ratio clean={float(r[-1][sink]):.2f} killed={float(rk[-1][sink]):.2f}; "
                  f"top-norm patch killed={int(rk[-1].argmax())}", flush=True)
        stats["sink_rank"].append(int((d > d[sink]).sum()))
        task_of.append(task)
        for v in variants:
            name, k = (v.split("@") + [None])[:2]
            feats = fk if name.startswith("kill_") else f
            name = {"kill_full": "full", "kill_minus_sink": "full_minus_sink"}.get(name, name)
            idx = select(name, int(k) if k else 0, d, rf, sink, rng)
            pred = vlm.answer(feats, idx.to(device), question)
            rows[v].append(int(correct(task, pred, gold)))
            if n_done < SMOKE:
                print(f"  smoke [{task}] {v:>18}: k={len(idx)} pred={pred!r} gold={gold!r}", flush=True)
        if len(cache) > 300:  # bound GPU memory: keep only the calibration images
            for kk in list(cache)[N_CAL:]:
                if kk != key:
                    del cache[kk]
        if (n_done + 1) % 100 == 0:
            el = time.time() - t1
            print(f"  {n_done + 1}/{len(items)} done, {el / (n_done + 1):.2f}s/item; running acc full="
                  f"{np.mean(rows['full']):.3f} kill={np.mean(rows['kill_full']):.3f} topk@16={np.mean(rows['topk@16']):.3f} "
                  f"random@16={np.mean(rows['random@16']):.3f}", flush=True)

    sr = np.array(stats["sink_rank"])
    print("\nPart 1 summary:", json.dumps({
        "sink_in_top64_share": float((sr < 64).mean()), "sink_in_top144_share": float((sr < 144).mean()),
        "sink_rank1_share": float((sr == 0).mean()), "sink_rank_median": float(np.median(sr))}), flush=True)
    print("\nPart 2 results (accuracy; paired bootstrap 95% CI of the difference):", flush=True)
    for task in ("pope", "gqa", "all"):
        m = [t == task or task == "all" for t in task_of]
        sub = {v: [x for x, keep in zip(rows[v], m) if keep] for v in variants}
        for v in variants:
            acc = float(np.mean(sub[v]))
            line = {"task": task, "variant": v, "n": len(sub[v]), "acc": acc}
            if v != "full":
                line["vs_full"] = boot(sub[v], sub["full"])
            if "@" in v and not v.startswith("topk@"):
                line["vs_topk"] = boot(sub[v], sub["topk@" + v.split("@")[1]])
            if v == "kill_minus_sink":
                line["vs_full_minus_sink"] = boot(sub[v], sub["full_minus_sink"])
            print("RESULT_JSON " + json.dumps(line), flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
