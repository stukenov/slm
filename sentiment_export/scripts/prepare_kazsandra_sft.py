"""Convert KazSAnDRA dataset to binary sentiment SFT format.

Maps scores 1-2 → negative, 4-5 → positive, drops 3 (neutral).
Balances classes by undersampling the majority class.

Output format: {"text": "review text", "label": "positive"/"negative"}
"""

import argparse
import random

from datasets import Dataset, DatasetDict, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Local save path")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to")
    parser.add_argument("--balance", action="store_true", help="Undersample majority class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading issai/kazsandra...")
    ds = load_dataset("issai/kazsandra")["train"]

    # Map scores to binary labels
    positives = []
    negatives = []
    for ex in ds:
        text = ex["text"].strip()
        if not text:
            continue
        if ex["label"] in (4, 5):
            positives.append({"text": text, "label": "positive"})
        elif ex["label"] in (1, 2):
            negatives.append({"text": text, "label": "negative"})
        # label == 3 → skip

    print(f"Positive: {len(positives)}, Negative: {len(negatives)}")

    # Balance by undersampling
    if args.balance:
        min_count = min(len(positives), len(negatives))
        random.shuffle(positives)
        random.shuffle(negatives)
        positives = positives[:min_count]
        negatives = negatives[:min_count]
        print(f"Balanced to {min_count} per class")

    all_examples = positives + negatives
    random.shuffle(all_examples)

    # Split train/val
    val_size = int(len(all_examples) * args.val_ratio)
    val_data = all_examples[:val_size]
    train_data = all_examples[val_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
    })

    if args.output:
        dataset.save_to_disk(args.output)
        print(f"Saved to {args.output}")

    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub)
        print(f"Pushed to {args.push_to_hub}")

    if not args.output and not args.push_to_hub:
        print("Preview (first 5):")
        for ex in train_data[:5]:
            print(f"  [{ex['label']}] {ex['text'][:80]}")


if __name__ == "__main__":
    main()
