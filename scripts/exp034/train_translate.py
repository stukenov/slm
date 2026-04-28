"""
exp034: Continue-pretrain EkiTil models for kk↔ru translation.

Format: <|kk|> казахский текст <|translate|> <|ru|> русский перевод
        <|ru|> русский текст <|translate|> <|kk|> казахский перевод

Usage:
  # Full finetune on 1 GPU
  python3 train_translate.py --base stukenov/ekitil-core-qwen3-123m-kkru-base-v1 --epochs 3

  # With LoRA
  python3 train_translate.py --base stukenov/ekitil-core-qwen3-600m-kkru-base-v1 --lora --lora-r 32

  # Resume from checkpoint
  python3 train_translate.py --base stukenov/ekitil-core-qwen3-300m-kkru-base-v1 --resume /path/to/ckpt
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, math, time, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- Special tokens ----
KK_TAG = "<|kk|>"
RU_TAG = "<|ru|>"
TRANSLATE_TAG = "<|translate|>"

# ---- Telegram notifications ----
TG_BOT_TOKEN = "8620178354:AAHRfG7FX4bIqK-Hq_W7XoGCaB7FI6MKFb8"
TG_CHAT_ID = "47474471"

def tg_send(msg):
    import urllib.request, urllib.parse
    try:
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TG_CHAT_ID, "text": msg}).encode()
        urllib.request.urlopen(url, data, timeout=5)
    except Exception:
        pass


# ============================================================
# Dataset
# ============================================================
class TranslationDataset(Dataset):
    """Parallel pairs formatted for causal LM translation training."""

    def __init__(self, tokenizer, max_len=512, split="train", max_pairs=None):
        from datasets import load_dataset

        print("Loading parallel data...")
        ds = load_dataset("stukenov/ekitil-parallel-kkru-v2", data_dir="kk-ru", split=split)
        if max_pairs:
            ds = ds.select(range(min(max_pairs, len(ds))))

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []

        # Get special token IDs
        self.kk_id = tokenizer.convert_tokens_to_ids(KK_TAG)
        self.ru_id = tokenizer.convert_tokens_to_ids(RU_TAG)
        self.tr_id = tokenizer.convert_tokens_to_ids(TRANSLATE_TAG)

        if self.kk_id == tokenizer.unk_token_id:
            raise ValueError(f"Token {KK_TAG} not in tokenizer vocab! Add special tokens first.")

        # Pre-tokenize all pairs in both directions
        print(f"Tokenizing {len(ds)} pairs (both directions)...")
        skipped = 0
        for row in ds:
            kk_text = row["kk"].strip()
            ru_text = row["ru"].strip()
            if not kk_text or not ru_text:
                skipped += 1
                continue

            # kk → ru direction
            kk_to_ru = self._format_pair(kk_text, ru_text, src_lang="kk")
            if kk_to_ru is not None:
                self.pairs.append(kk_to_ru)

            # ru → kk direction
            ru_to_kk = self._format_pair(ru_text, kk_text, src_lang="ru")
            if ru_to_kk is not None:
                self.pairs.append(ru_to_kk)

        print(f"  Total training examples: {len(self.pairs):,} (skipped {skipped})")

    def _format_pair(self, src_text, tgt_text, src_lang):
        """Format: <|src|> src_text <|translate|> <|tgt|> tgt_text <eos>"""
        src_tag = KK_TAG if src_lang == "kk" else RU_TAG
        tgt_tag = RU_TAG if src_lang == "kk" else KK_TAG

        # Tokenize source and target separately
        src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=False)

        src_tag_id = self.kk_id if src_lang == "kk" else self.ru_id
        tgt_tag_id = self.ru_id if src_lang == "kk" else self.kk_id

        # Full sequence: [src_tag] + src_ids + [translate_tag] + [tgt_tag] + tgt_ids + [eos]
        eos_id = self.tokenizer.eos_token_id or 0
        full_ids = [src_tag_id] + src_ids + [self.tr_id] + [tgt_tag_id] + tgt_ids + [eos_id]

        if len(full_ids) > self.max_len:
            return None

        # Labels: mask source side (only predict target)
        # Source = [src_tag] + src_ids + [translate_tag] + [tgt_tag]
        src_len = 1 + len(src_ids) + 1 + 1  # src_tag + src + translate + tgt_tag
        labels = [-100] * src_len + tgt_ids + [eos_id]

        assert len(full_ids) == len(labels), f"Mismatch: {len(full_ids)} vs {len(labels)}"

        return {
            "input_ids": full_ids,
            "labels": labels,
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    """Pad batch to max length."""
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids = []
    labels = []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ============================================================
# Training
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    # Check if special tokens exist
    special_tokens = [KK_TAG, RU_TAG, TRANSLATE_TAG]
    missing = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if missing:
        print(f"Adding missing special tokens: {missing}")
        tokenizer.add_special_tokens({"additional_special_tokens": missing})

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Resize embeddings if tokens were added
    if missing:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings to {len(tokenizer)}")

    # LoRA setup
    if args.lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable parameters: {n_params:.1f}M")

    # Dataset
    dataset = TranslationDataset(
        tokenizer, max_len=args.max_len, max_pairs=args.max_pairs,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # LR scheduler (cosine with warmup)
    total_steps = (len(dataloader) // args.grad_accum) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    save_steps = max(len(dataloader) // 4, 100)  # Save 4 times per epoch
    log_steps = 10
    best_loss = float("inf")
    global_step = 0

    model_name = args.base.split("/")[-1].replace("base", "translate")
    tg_send(f"🚀 exp034 started: {model_name}\n"
            f"Pairs: {len(dataset):,}, Epochs: {args.epochs}\n"
            f"LoRA: {args.lora}, LR: {args.lr}")

    print(f"\nTraining: {len(dataset):,} examples, {args.epochs} epochs, {total_steps} steps")
    print(f"Save every {save_steps} steps to {args.output_dir}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        t0 = time.time()

        for step, batch in enumerate(dataloader):
            global_step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Update LR
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.grad_accum

            # Backward
            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Stats (loss was divided by grad_accum, undo for logging)
            n_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * args.grad_accum * n_tokens
            epoch_tokens += n_tokens

            if global_step % log_steps == 0:
                avg_loss = epoch_loss / max(1, epoch_tokens)
                elapsed = time.time() - t0
                tok_s = epoch_tokens / max(0.1, elapsed)
                eta_h = (total_steps - global_step) * elapsed / max(1, step + 1) / 3600
                print(f"  step {global_step:>6}/{total_steps} | loss {avg_loss:.4f} | "
                      f"lr {lr:.2e} | {tok_s:.0f} tok/s | epoch {epoch + (step+1)/len(dataloader):.2f} | "
                      f"ETA {eta_h:.1f}h", flush=True)

            # Save checkpoint
            if global_step % save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                if args.lora:
                    model.save_pretrained(ckpt_dir)
                else:
                    model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                avg_loss = epoch_loss / max(1, epoch_tokens)
                with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                    json.dump({"step": global_step, "epoch": epoch, "loss": avg_loss, "lr": lr}, f)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                tg_send(f"💾 step {global_step}/{total_steps} | loss {avg_loss:.4f} | epoch {epoch}")

        avg_loss = epoch_loss / max(1, epoch_tokens)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, tokens={epoch_tokens:,}")
        tg_send(f"📊 Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    if args.lora:
        model.save_pretrained(final_dir)
    else:
        model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "meta.json"), "w") as f:
        json.dump({
            "step": global_step, "epochs": args.epochs,
            "loss": avg_loss, "base_model": args.base,
            "lora": args.lora, "lora_r": args.lora_r if args.lora else None,
        }, f)

    # Upload to HF
    if args.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.hf_repo, exist_ok=True, private=False)

        if args.lora:
            # For LoRA, merge and upload
            print("Merging LoRA weights...")
            merged = model.merge_and_unload()
            merged.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)

        api.upload_folder(folder_path=final_dir, repo_id=args.hf_repo)
        print(f"Uploaded to {args.hf_repo}")
        tg_send(f"✅ DONE! Uploaded to {args.hf_repo}\nFinal loss: {avg_loss:.4f}")

    print(f"\nTraining complete! Final loss: {avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model HF repo or path")
    parser.add_argument("--output-dir", default="/root/checkpoints/translate")
    parser.add_argument("--hf-repo", default=None, help="HF repo to upload final model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit pairs for testing")
    parser.add_argument("--lora", action="store_true", help="Use LoRA instead of full finetune")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Auto-generate HF repo name
    if not args.hf_repo:
        base_name = args.base.split("/")[-1]
        suffix = "-lora" if args.lora else ""
        args.hf_repo = f"stukenov/{base_name.replace('base', 'translate')}{suffix}"

    print(f"Config:")
    print(f"  Base: {args.base}")
    print(f"  Output: {args.output_dir}")
    print(f"  HF repo: {args.hf_repo}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Max len: {args.max_len}")
    print(f"  LoRA: {args.lora} (r={args.lora_r})")
    print()

    train(args)


if __name__ == "__main__":
    main()
