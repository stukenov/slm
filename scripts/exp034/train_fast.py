"""exp034: Fast translation training from pretokenized cache (local trusted pickle).
Uses numpy storage, torch.compile, large batches, multi-worker dataloader."""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math, time, json, argparse, pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request, urllib.parse

TG_BOT = "8620178354:AAHRfG7FX4bIqK-Hq_W7XoGCaB7FI6MKFb8"
TG_CHAT = "47474471"

def tg(msg):
    try:
        d = urllib.parse.urlencode({"chat_id": TG_CHAT, "text": msg}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{TG_BOT}/sendMessage", d, timeout=5)
    except Exception:
        pass


class TranslateDataset(Dataset):
    def __init__(self, path):
        t0 = time.time()
        print(f"Loading {path}...", flush=True)
        with open(path, "rb") as f:
            raw = pickle.load(f)  # trusted local cache
        self.data = [(np.array(ids, dtype=np.int32), np.array(lab, dtype=np.int32))
                     for ids, lab in zip(raw["input_ids"], raw["labels"])]
        del raw
        print(f"  {len(self.data):,} examples in {time.time()-t0:.1f}s", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, lab = self.data[idx]
        return torch.from_numpy(ids).long(), torch.from_numpy(lab).long()


def collate(batch):
    max_len = max(ids.shape[0] for ids, _ in batch)
    bs = len(batch)
    input_ids = torch.zeros(bs, max_len, dtype=torch.long)
    labels = torch.full((bs, max_len), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        input_ids[i, :ids.shape[0]] = ids
        labels[i, :lab.shape[0]] = lab
    return input_ids, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="stukenov/ekitil-core-qwen3-600m-kkru-base-v1")
    p.add_argument("--cache", default="/workspace/cache/data.pkl")
    p.add_argument("--tokenizer", default="/workspace/cache/tokenizer")
    p.add_argument("--out", default="/workspace/checkpoints/translate")
    p.add_argument("--hf", default="stukenov/ekitil-core-qwen3-600m-kkru-translate-v1")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()

    dev = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    print(f"Loading {a.base}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    if len(tok) != model.config.vocab_size:
        model.resize_token_embeddings(len(tok))
        print(f"Resized vocab -> {len(tok)}", flush=True)
    model = model.to(dev)
    npar = sum(x.numel() for x in model.parameters()) / 1e6
    print(f"Params: {npar:.1f}M", flush=True)

    if a.compile:
        print("torch.compile()...", flush=True)
        model = torch.compile(model)

    ds = TranslateDataset(a.cache)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate,
                    num_workers=8, pin_memory=True, persistent_workers=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)

    spe = len(dl) // a.accum
    total = spe * a.epochs
    warmup = min(a.warmup, total // 10)
    save_every = max(spe // 4, 500)

    def lr_fn(s):
        if s < warmup:
            return a.lr * s / max(1, warmup)
        prog = (s - warmup) / max(1, total - warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * prog))

    os.makedirs(a.out, exist_ok=True)
    eff_bs = a.bs * a.accum
    print(f"\nbs={a.bs}x{a.accum}={eff_bs}, {spe:,} steps/ep, {total:,} total, save every {save_every}", flush=True)
    tg(f"exp034: {npar:.0f}M, {len(ds):,}ex, {total:,}steps, bs={eff_bs}")

    gstep = 0
    for ep in range(a.epochs):
        model.train()
        eloss = etok = 0
        t0 = time.time()

        for i, (ids, lab) in enumerate(dl):
            ids, lab = ids.to(dev), lab.to(dev)
            ostep = gstep // a.accum
            lr = lr_fn(ostep)
            for pg in opt.param_groups:
                pg["lr"] = lr

            out = model(input_ids=ids, labels=lab)
            loss = out.loss / a.accum
            loss.backward()

            nt = (lab != -100).sum().item()
            eloss += loss.item() * a.accum * nt
            etok += nt
            gstep += 1

            if gstep % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            if gstep % (25 * a.accum) == 0:
                al = eloss / max(1, etok)
                el = time.time() - t0
                tps = etok / max(0.1, el)
                eta = (total * a.accum - gstep) * el / max(1, gstep - ep * len(dl)) / 3600
                epc = ep + (i + 1) / len(dl)
                print(f"  step {ostep:>7,}/{total:,} | loss {al:.4f} | lr {lr:.2e} | {tps:.0f} tok/s | ep {epc:.2f} | ETA {eta:.1f}h", flush=True)

            if gstep % (save_every * a.accum) == 0:
                cd = os.path.join(a.out, f"step_{ostep}")
                os.makedirs(cd, exist_ok=True)
                m = model._orig_mod if hasattr(model, "_orig_mod") else model
                m.save_pretrained(cd)
                tok.save_pretrained(cd)
                al = eloss / max(1, etok)
                epc = ep + (i + 1) / len(dl)
                with open(os.path.join(cd, "meta.json"), "w") as f:
                    json.dump({"step": ostep, "epoch": ep, "loss": al}, f)
                tg(f"step {ostep}/{total} | loss {al:.4f} | ep {epc:.2f}")
                print(f"  Saved: {cd}", flush=True)

        al = eloss / max(1, etok)
        print(f"Epoch {ep+1}/{a.epochs}: loss={al:.4f}", flush=True)
        tg(f"Epoch {ep+1}/{a.epochs}: loss={al:.4f}")

    fd = os.path.join(a.out, "final")
    os.makedirs(fd, exist_ok=True)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    m.save_pretrained(fd)
    tok.save_pretrained(fd)
    al = eloss / max(1, etok)
    with open(os.path.join(fd, "meta.json"), "w") as f:
        json.dump({"step": total, "epochs": a.epochs, "loss": al, "base": a.base}, f)

    if a.hf:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.hf, exist_ok=True, private=False)
        api.upload_folder(folder_path=fd, repo_id=a.hf)
        print(f"Uploaded: {a.hf}", flush=True)
        tg(f"DONE! {a.hf} loss={al:.4f}")

    print(f"\nTraining complete! loss={al:.4f}", flush=True)

if __name__ == "__main__":
    main()
