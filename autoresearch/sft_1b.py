"""SFT fine-tuning of 1.08B Kazakh Llama base model into instruct version.
Loads from HF checkpoint branch, fine-tunes on ChatML instruction data, publishes result.

Usage: python sft_1b.py [--base-revision step-16000] [--epochs 3] [--lr 2e-5]
"""
import os, sys, json, time, argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from huggingface_hub import hf_hub_download, HfApi, create_repo
from datasets import load_dataset

def send_tg(msg):
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({"chat_id": "47474471", "text": msg}).encode()
        urllib.request.urlopen("https://api.telegram.org/bot8620178354:AAFFqHqTvgobauCLiJ61CO1clWKG-CO-K1g/sendMessage", data, timeout=10)
    except Exception:
        pass

# Simple prompt format (no special tokens needed)
def format_prompt(messages):
    """Convert messages to simple prompt format."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"### Сұрақ:\n{content}\n\n### Жауап:\n"
        elif role == "assistant":
            text += f"{content}\n\n"
    return text

class SFTDataset(Dataset):
    def __init__(self, tokenizer, max_len=1024):
        print("Loading instruction dataset (AmanMussa/kazakh-instruction-v2)...")
        ds = load_dataset("AmanMussa/kazakh-instruction-v2", split="train")
        print(f"  Loaded {len(ds)} examples")

        self.examples = []
        skipped = 0
        for row in ds:
            instruction = row.get("instruction", "")
            inp = row.get("input", "")
            output = row.get("output", "")
            if not instruction or not output:
                skipped += 1
                continue
            # Build ChatML from instruction/input/output
            user_msg = instruction
            if inp:
                user_msg += f"\n{inp}"
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
            text = format_prompt(messages)
            ids = tokenizer.encode(text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            if len(ids) < 10:
                skipped += 1
                continue
            self.examples.append(torch.tensor(ids, dtype=torch.long))
        print(f"  {len(self.examples)} valid examples, {skipped} skipped")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return ids[:-1], ids[1:]  # input, target

def collate_fn(batch):
    """Pad batch to same length."""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), -100, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :inp.size(0)] = inp
        padded_targets[i, :tgt.size(0)] = tgt
    return padded_inputs, padded_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-revision", default="step-16000")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--repo", default="stukenov/sozkz-core-llama-1b-kk-instruct-v1")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    device = "cuda:0"

    send_tg(f"[SFT] Starting instruct fine-tuning from {args.base_revision}, {args.epochs} epochs, lr={args.lr}")

    # Load base model
    print(f"Loading base model from revision={args.base_revision}...")
    model = LlamaForCausalLM.from_pretrained(
        "stukenov/sozkz-core-llama-1b-kk-base-v1",
        revision=args.base_revision,
        dtype=torch.bfloat16,
        device_map=device,
    )
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Load tokenizer
    tok_file = hf_hub_download("stukenov/sozkz-core-gpt2-50k-kk-base-v1", "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 1

    # No special tokens needed — using simple prompt format

    # Dataset
    dataset = SFTDataset(tokenizer, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs // args.grad_accum
    warmup_steps = min(100, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"  Epochs: {args.epochs}, Steps/epoch: {len(loader)}, Total optim steps: {total_steps}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    send_tg(f"[SFT] {len(dataset)} examples, {total_steps} steps, effective batch={args.batch_size * args.grad_accum}")

    # Training
    model.train()
    t0 = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=inputs, labels=targets)
                loss = outputs.loss / args.grad_accum

            loss.backward()
            epoch_loss += outputs.loss.item()
            epoch_tokens += (targets != -100).sum().item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = epoch_loss / (step + 1)
                    elapsed = time.time() - t0
                    tps = epoch_tokens / elapsed
                    print(f"  epoch {epoch+1}/{args.epochs} step {global_step}/{total_steps} | loss {avg_loss:.4f} | {tps:.0f} tok/s")

        avg_epoch_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1} done: loss={avg_epoch_loss:.4f}, {elapsed/60:.1f}min")
        send_tg(f"[SFT] Epoch {epoch+1}/{args.epochs} done: loss={avg_epoch_loss:.4f}, {elapsed/60:.1f}min")

    total_time = time.time() - t0
    print(f"\nSFT done in {total_time/60:.1f}min")

    # Inference test
    print("\n=== Inference test ===")
    model.eval()
    test_questions = [
        "Қазақстанның астанасы қай қала?",
        "Абай Құнанбаев туралы қысқаша айтып бер",
        "Питон тілінде сөздікті қалай сұрыптауға болады?",
    ]
    inference_results = []
    for q in test_questions:
        prompt = f"### Сұрақ:\n{q}\n\n### Жауап:\n"
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        resp = text.split("### Жауап:\n")[-1].split("### Сұрақ:")[0].strip()
        print(f"\nQ: {q}")
        print(f"A: {resp[:300]}")
        inference_results.append(f"Q: {q}\nA: {resp[:300]}")

    # Publish
    print(f"\nPublishing to {args.repo}...")
    create_repo(args.repo, token=token, exist_ok=True)
    model.push_to_hub(args.repo, token=token)
    tokenizer.push_to_hub(args.repo, token=token)

    # Upload results
    api = HfApi(token=token)
    results = {
        "base_model": "stukenov/sozkz-core-llama-1b-kk-base-v1",
        "base_revision": args.base_revision,
        "sft_dataset": "AmanMussa/kazakh-instruction-v2",
        "sft_examples": len(dataset),
        "epochs": args.epochs,
        "lr": args.lr,
        "final_loss": avg_epoch_loss,
        "training_minutes": round(total_time / 60, 1),
        "inference_test": inference_results,
    }
    with open("/tmp/sft_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    api.upload_file(path_or_fileobj="/tmp/sft_results.json", path_in_repo="sft_results.json", repo_id=args.repo, token=token)

    print(f"\nDone! https://huggingface.co/{args.repo}")
    send_tg(f"[SFT] DONE! Instruct model published: https://huggingface.co/{args.repo}\nFinal loss: {avg_epoch_loss:.4f}\nTime: {total_time/60:.1f}min\n\n" + "\n\n".join(inference_results))

if __name__ == "__main__":
    main()
