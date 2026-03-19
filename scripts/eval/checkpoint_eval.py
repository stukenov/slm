"""Quick checkpoint evaluation: generate text samples and measure quality."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

KAZAKH_SPECIFIC = set("ӘәҒғҚқҢңӨөҰұҮүҺһІі")
CYRILLIC_RANGE = re.compile(r"[\u0400-\u04FF]")


def kazakh_char_ratio(text: str) -> float:
    cyrillic_chars = CYRILLIC_RANGE.findall(text)
    if not cyrillic_chars:
        return 0.0
    kazakh_count = sum(1 for c in cyrillic_chars if c in KAZAKH_SPECIFIC)
    return kazakh_count / len(cyrillic_chars)


def load_prompts(path: str) -> list[str]:
    lines = Path(path).read_text().strip().split("\n")
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def run_eval(
    checkpoint_path: str,
    tokenizer_path: str,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> dict:
    logger.info("Loading model from %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # Read trainer_state for loss info
    trainer_state_path = Path(checkpoint_path) / "trainer_state.json"
    train_loss = None
    global_step = None
    if trainer_state_path.exists():
        state = json.loads(trainer_state_path.read_text())
        global_step = state.get("global_step")
        log_history = state.get("log_history", [])
        for entry in reversed(log_history):
            if "loss" in entry:
                train_loss = entry["loss"]
                break

    results = {
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
        "global_step": global_step,
        "train_loss": train_loss,
        "parameters_m": round(params_m, 2),
        "generations": [],
    }

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        kk_ratio = kazakh_char_ratio(text)
        results["generations"].append({
            "prompt": prompt,
            "generated": text,
            "kazakh_char_ratio": round(kk_ratio, 4),
        })
        logger.info("[Prompt] %s", prompt)
        logger.info("[Output] %s", text[:200])
        logger.info("")

    avg_kk = sum(g["kazakh_char_ratio"] for g in results["generations"]) / len(results["generations"])
    results["avg_kazakh_char_ratio"] = round(avg_kk, 4)

    return results


def main():
    parser = argparse.ArgumentParser(description="Quick checkpoint eval")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (defaults to checkpoint)")
    parser.add_argument("--prompts", type=str, default="eval/prompts_kk.txt")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.checkpoint
    prompts = load_prompts(args.prompts)

    results = run_eval(
        checkpoint_path=args.checkpoint,
        tokenizer_path=tokenizer_path,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        step = results.get("global_step", "unknown")
        out_path = Path("eval") / f"checkpoint_{step}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Results saved to %s", out_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Step: {results['global_step']}, Train Loss: {results['train_loss']}")
    print(f"Avg Kazakh char ratio: {results['avg_kazakh_char_ratio']:.4f}")
    print(f"{'='*60}")
    for g in results["generations"]:
        print(f"\n[{g['prompt']}]")
        print(f"  -> {g['generated'][:150]}")
    print()


if __name__ == "__main__":
    main()
