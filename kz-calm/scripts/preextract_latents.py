"""Pre-extract Mimi 512-dim latents from codes dataset. Saves to disk as .pt shards."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import argparse
import logging
import time
import torch
from datasets import load_dataset
from kzcalm.model.code_embedding import MimiLatentExtractor

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/workspace/latents_512")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading Mimi...")
    extractor = MimiLatentExtractor(device=device)

    logger.info("Streaming dataset...")
    ds = load_dataset("stukenov/kzcalm-mimi-codes-kk-v1", split="train", streaming=True)

    shard_idx = 0
    shard_data = []
    total = 0
    t0 = time.time()

    for sample in ds:
        if args.max_samples > 0 and total >= args.max_samples:
            break

        num_frames = sample["num_frames"]
        if num_frames > 1500:
            continue

        codes = torch.tensor(sample["codes"][:8], dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            latents = extractor(codes)  # (1, 2T, 512)

        shard_data.append({
            "text": sample["text"],
            "latents": latents.squeeze(0).half().cpu(),  # (2T, 512) float16
            "num_latent_frames": latents.shape[1],
        })
        total += 1

        # Save shard every 1000 samples
        if len(shard_data) >= 1000:
            path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
            torch.save(shard_data, path)
            elapsed = time.time() - t0
            speed = total / elapsed
            logger.info(f"Shard {shard_idx}: {total} samples, {speed:.0f} samples/s")
            shard_data = []
            shard_idx += 1

    # Save remaining
    if shard_data:
        path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt")
        torch.save(shard_data, path)
        shard_idx += 1

    elapsed = time.time() - t0
    logger.info(f"Done: {total} samples, {shard_idx} shards, {elapsed:.0f}s ({total/elapsed:.0f} samples/s)")


if __name__ == "__main__":
    main()
