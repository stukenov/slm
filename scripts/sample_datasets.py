"""Sample and display rows from 3 SFT datasets for quality assessment."""
import random, json
from datasets import load_dataset

random.seed(42)

datasets_info = [
    ("stukenov/sozkz-instruct-chatml-kk-v1", "chatml", 20),
    ("stukenov/sozkz-corpus-chatml-kk-instruct-mix-v1", "chatml", 20),
    ("AmanMussa/kazakh-instruction-v2", "alpaca", 20),
]

for ds_name, fmt, n in datasets_info:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"DATASET: {ds_name}")
    print(sep)
    ds = load_dataset(ds_name, split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i >= 5000:
            break

    samples = random.sample(rows, min(n, len(rows)))

    for idx, row in enumerate(samples):
        if fmt == "chatml":
            msgs = row.get("messages", [])
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            src = row.get("source", "?")
            nt = row.get("num_turns", "?")
            user_q = next((m["content"][:200] for m in msgs if m["role"] == "user"), "")
            asst_a = next((m["content"][:200] for m in msgs if m["role"] == "assistant"), "")
            print(f"\n[{idx+1}] src={src} turns={nt}")
            print(f"  Q: {user_q}")
            print(f"  A: {asst_a}")
        else:
            instr = row.get("instruction", "")[:200]
            inp = row.get("input", "")[:100]
            out = row.get("output", "")[:200]
            print(f"\n[{idx+1}]")
            print(f"  I: {instr}")
            if inp:
                print(f"  In: {inp}")
            print(f"  O: {out}")
