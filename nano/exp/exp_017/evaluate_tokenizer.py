#!/usr/bin/env python3
"""Evaluate and compare tokenizers on Kazakh text.

Metrics:
  - Fertility (tokens per word)
  - Compression ratio (bytes per token)
  - Morpheme alignment score
  - Suffix consistency score

Usage:
    python evaluate_tokenizer.py --morpheme ./output/morphbpe-rule-100k --baseline ./output/baseline-bpe-100k
    python evaluate_tokenizer.py --morpheme ./output/morphbpe-rule-100k --compare stukenov/sozkz-core-gpt2-100k-kk-base-v1
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from transformers import AutoTokenizer

from morpheme_segmenter import MorphemeSegmenter, MORPH_SEP


# --- Test data ---

# Words with known morpheme boundaries (gold standard)
GOLD_MORPHEMES = {
    "үйлерімізде":    ["үй", "лер", "іміз", "де"],
    "мектептегі":      ["мектеп", "тегі"],
    "оқушылар":        ["оқу", "шы", "лар"],
    "математиканы":    ["математика", "ны"],
    "Қазақстанның":    ["Қазақстан", "ның"],
    "қаласында":       ["қала", "сын", "да"],
    "жазылғандар":     ["жаз", "ыл", "ған", "дар"],
    "университеттерде": ["университет", "тер", "де"],
    "айтқанымызды":    ["айт", "қан", "ымыз", "ды"],
    "кітаптарымызда":  ["кітап", "тар", "ымыз", "да"],
    "болғандықтан":    ["бол", "ған", "дық", "тан"],
    "оқығандарымыз":   ["оқы", "ған", "дар", "ымыз"],
    "жұмысшыларға":    ["жұмыс", "шы", "лар", "ға"],
    "мұғалімдердің":   ["мұғалім", "дер", "дің"],
    "балаларымен":     ["бала", "лар", "ымен"],
}

# Sentences for fertility comparison
TEST_SENTENCES = [
    "Қазақстан — Орталық Азиядағы мемлекет.",
    "Бүгін ауа райы жақсы болады.",
    "2024 жылы халықаралық конференция өтеді.",
    "Мектепте оқушылар математика сабағына дайындалуда.",
    "Алматы қаласында жаңа метро стансасы ашылды.",
    "Үйлерімізде кітаптар көп.",
    "Университеттердегі студенттер емтихандарға дайындалуда.",
    "Ғылыми-зерттеу институтының қызметкерлері жаңалық ашты.",
    "Қазақ тілі — түркі тілдер тобына жатады.",
    "Биылғы жазда туристер саны артты.",
    "Экономикалық дамуға байланысты жаңа бағдарламалар қабылданды.",
    "Мәдениет министрлігі жаңа жобаларды қолдайды.",
    "Ауыл шаруашылығында заманауи технологиялар қолданылуда.",
    "Білім беру жүйесінде реформалар жүргізілуде.",
    "Ұлттық банк пайыздық мөлшерлемені өзгертті.",
]

# Suffixes to check for consistency
SUFFIX_TEST_GROUPS = {
    "plural": {
        "лар": ["балалар", "адамдар", "мұғалімдер", "кітаптар", "жұмысшылар"],
        "лер": ["үйлер", "көшелер", "мектептер", "университеттер", "мәселелер"],
        "дар": ["қаладар", "аңдар", "тоғандар"],
        "дер": ["жердер", "бөлмедер"],
        "тар": ["достар", "сөздіктар"],
        "тер": ["көліктер", "сағаттер"],
    },
    "genitive": {
        "ның": ["Қазақстанның", "адамның", "баланың"],
        "нің": ["үйдің", "мектептің", "көшенің"],
    },
    "locative": {
        "да": ["қалада", "мектепте", "үйде"],
        "де": ["көшеде", "бөлмеде", "далада"],
    },
}


def load_tokenizer(path_or_name: str):
    """Load a tokenizer from local path or HuggingFace."""
    try:
        return AutoTokenizer.from_pretrained(path_or_name)
    except Exception as e:
        print(f"ERROR loading {path_or_name}: {e}")
        return None


def measure_fertility(tokenizer, sentences: list[str]) -> dict:
    """Measure average tokens per word."""
    total_tokens = 0
    total_words = 0
    per_sentence = []

    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=False)
        words = len(sent.split())
        total_tokens += len(ids)
        total_words += words
        per_sentence.append({
            "text": sent,
            "tokens": len(ids),
            "words": words,
            "fertility": len(ids) / words if words > 0 else 0,
        })

    return {
        "avg_fertility": total_tokens / total_words if total_words > 0 else 0,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "per_sentence": per_sentence,
    }


def measure_compression(tokenizer, sentences: list[str]) -> float:
    """Measure bytes per token."""
    total_bytes = 0
    total_tokens = 0
    for sent in sentences:
        total_bytes += len(sent.encode("utf-8"))
        total_tokens += len(tokenizer.encode(sent, add_special_tokens=False))
    return total_bytes / total_tokens if total_tokens > 0 else 0


def measure_morpheme_alignment(tokenizer, gold: dict[str, list[str]]) -> dict:
    """Measure how well token boundaries align with morpheme boundaries."""
    total_boundaries = 0
    aligned_boundaries = 0
    results = []

    for word, morphemes in gold.items():
        ids = tokenizer.encode(word, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        # Decode each token to get character spans
        decoded_tokens = []
        for t in tokens:
            # Handle byte-level tokens (Ġ prefix = space, Ã etc = bytes)
            decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids([t]))
            decoded_tokens.append(decoded)

        # Build character position map for token boundaries
        token_boundaries = set()
        pos = 0
        for dt in decoded_tokens:
            pos += len(dt)
            token_boundaries.add(pos)

        # Build character position map for morpheme boundaries
        morph_boundaries = set()
        pos = 0
        for m in morphemes[:-1]:  # Last boundary is end of word, skip
            pos += len(m)
            morph_boundaries.add(pos)

        # Count aligned boundaries
        n_morph = len(morph_boundaries)
        n_aligned = len(morph_boundaries & token_boundaries)
        total_boundaries += n_morph
        aligned_boundaries += n_aligned

        results.append({
            "word": word,
            "morphemes": morphemes,
            "tokens": decoded_tokens,
            "morph_boundaries": sorted(morph_boundaries),
            "token_boundaries": sorted(token_boundaries),
            "aligned": n_aligned,
            "total": n_morph,
        })

    alignment = aligned_boundaries / total_boundaries if total_boundaries > 0 else 0
    return {
        "alignment_score": alignment,
        "aligned_boundaries": aligned_boundaries,
        "total_boundaries": total_boundaries,
        "per_word": results,
    }


def measure_suffix_consistency(tokenizer, suffix_groups: dict) -> dict:
    """Measure whether the same suffix is tokenized consistently across words."""
    results = {}
    total_consistent = 0
    total_groups = 0

    for category, suffixes in suffix_groups.items():
        cat_results = {}
        for suffix, words in suffixes.items():
            # For each word, check how the suffix portion is tokenized
            suffix_tokenizations = []
            for word in words:
                ids = tokenizer.encode(word, add_special_tokens=False)
                tokens = tokenizer.convert_ids_to_tokens(ids)
                suffix_tokenizations.append({
                    "word": word,
                    "tokens": tokens,
                    "n_tokens": len(tokens),
                })

            # Check consistency: do all words produce the same number of tokens?
            n_tokens_set = set(st["n_tokens"] for st in suffix_tokenizations)
            is_consistent = len(n_tokens_set) <= 1
            if is_consistent:
                total_consistent += 1
            total_groups += 1

            cat_results[suffix] = {
                "tokenizations": suffix_tokenizations,
                "consistent": is_consistent,
            }
        results[category] = cat_results

    return {
        "consistency_score": total_consistent / total_groups if total_groups > 0 else 0,
        "consistent_groups": total_consistent,
        "total_groups": total_groups,
        "details": results,
    }


def print_comparison(results: dict[str, dict]):
    """Print a comparison table of all tokenizers."""
    print(f"\n{'='*80}")
    print("TOKENIZER COMPARISON")
    print(f"{'='*80}")

    # Header
    names = list(results.keys())
    header = f"{'Metric':<30}" + "".join(f"{n:>20}" for n in names)
    print(header)
    print("-" * len(header))

    # Metrics
    metrics = [
        ("Fertility (tok/word)", "fertility", "avg_fertility"),
        ("Compression (bytes/tok)", "compression", None),
        ("Morpheme alignment", "alignment", "alignment_score"),
        ("Suffix consistency", "consistency", "consistency_score"),
    ]

    for label, key, subkey in metrics:
        row = f"{label:<30}"
        for name in names:
            val = results[name].get(key)
            if val is None:
                row += f"{'N/A':>20}"
            elif subkey:
                row += f"{val[subkey]:>20.3f}"
            else:
                row += f"{val:>20.3f}"
        print(row)

    print(f"\n{'='*80}")

    # Detailed fertility per sentence
    print("\nPer-sentence fertility:")
    print(f"{'Sentence':<50}" + "".join(f"{n:>15}" for n in names))
    print("-" * (50 + 15 * len(names)))

    for i, sent in enumerate(TEST_SENTENCES[:5]):
        row = f"{sent[:48]:<50}"
        for name in names:
            fert = results[name]["fertility"]["per_sentence"][i]
            row += f"{fert['fertility']:>15.2f}"
        print(row)

    # Morpheme alignment details
    print("\nMorpheme alignment details:")
    print(f"{'Word':<25}" + "".join(f"{n:>25}" for n in names))
    print("-" * (25 + 25 * len(names)))

    for word in list(GOLD_MORPHEMES.keys())[:8]:
        row = f"{word:<25}"
        for name in names:
            align = results[name]["alignment"]
            if align:
                for pw in align["per_word"]:
                    if pw["word"] == word:
                        tokens_str = " ".join(pw["tokens"])[:22]
                        row += f"{tokens_str:>25}"
                        break
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizers on Kazakh")
    parser.add_argument("--morpheme", type=str, help="Path to morpheme-aware tokenizer")
    parser.add_argument("--baseline", type=str, help="Path to baseline tokenizer")
    parser.add_argument("--compare", type=str, nargs="*", default=[],
                        help="Additional tokenizers to compare (HF names or paths)")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    tokenizers = {}

    if args.morpheme:
        tok = load_tokenizer(args.morpheme)
        if tok:
            tokenizers["morpheme-bpe"] = tok

    if args.baseline:
        tok = load_tokenizer(args.baseline)
        if tok:
            tokenizers["baseline-bpe"] = tok

    for name_or_path in args.compare:
        tok = load_tokenizer(name_or_path)
        if tok:
            short_name = name_or_path.split("/")[-1][:20]
            tokenizers[short_name] = tok

    if not tokenizers:
        print("No tokenizers loaded. Provide --morpheme, --baseline, or --compare.")
        return

    # Evaluate each tokenizer
    results = {}
    for name, tok in tokenizers.items():
        print(f"\nEvaluating: {name}")
        results[name] = {
            "fertility": measure_fertility(tok, TEST_SENTENCES),
            "compression": measure_compression(tok, TEST_SENTENCES),
            "alignment": measure_morpheme_alignment(tok, GOLD_MORPHEMES),
            "consistency": measure_suffix_consistency(tok, SUFFIX_TEST_GROUPS),
        }

    # Print comparison
    print_comparison(results)

    # Save results
    if args.output:
        # Convert sets to lists for JSON serialization
        def serialize(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=serialize)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
