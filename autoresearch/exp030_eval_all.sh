#!/bin/bash
# ============================================================================
# exp030: Evaluate all experiment checkpoints and print comparison table
# ============================================================================
set -e

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")
export TG_TOKEN="8620178354:AAGVRqNqAaGM_6JN_dQHbTWEbVsBvqJB6Xk"
export TG_CHAT_ID="47474471"

cd /root
mkdir -p exp030_results

EXPS=""

# Find all experiment dirs
for dir in exp030_030*/final; do
    if [ -d "$dir" ]; then
        exp_id=$(echo "$dir" | sed 's|exp030_||;s|/final||')
        echo "=== Evaluating $exp_id ==="

        # Detect method
        if [ -f "$dir/adapter_config.json" ]; then
            METHOD="lora"
        else
            METHOD="full"
        fi

        CUDA_VISIBLE_DEVICES=0 python3 exp030_eval.py \
            --model_dir "$dir" \
            --exp_id "$exp_id" \
            --method "$METHOD" \
            --max_examples 500 \
            --output_dir exp030_results \
            2>&1 | tail -20

        EXPS="$EXPS $exp_id"
        echo ""
    fi
done

# Build comparison table
echo ""
echo "Building comparison table..."
python3 << 'TABLEEOF'
import json, os, glob

results = []
for f in sorted(glob.glob("exp030_results/eval_*.json")):
    with open(f) as fh:
        data = json.load(fh)
    m = data["metrics"]
    exp_id = data["exp_id"]

    # Load config from results.json if available
    config = {}
    results_file = f"/root/exp030_{exp_id}/final/results.json"
    if os.path.exists(results_file):
        with open(results_file) as cf:
            config = json.load(cf)

    results.append({
        "exp_id": exp_id,
        "method": config.get("method", "?"),
        "rank": config.get("lora_rank", "-"),
        "clean": config.get("clean_ratio", "?"),
        "lr": config.get("lr", "?"),
        "epochs": config.get("epochs", "?"),
        "em": m["exact_match"],
        "cer": m["cer"],
        "f05": m["word_f05"],
        "id_pres": m.get("identity_preserved", "-"),
        "eval_loss": config.get("eval_loss", "-"),
    })

# Print table
header = f"| {'Exp':>8} | {'Method':>6} | {'Rank':>4} | {'Clean':>5} | {'LR':>7} | {'Ep':>2} | {'EM%':>5} | {'CER':>6} | {'F0.5':>5} | {'ID%':>5} | {'Loss':>6} |"
sep = "|" + "-" * (len(header) - 2) + "|"
print(f"\n{'='*len(header)}")
print("EXP030 GEC 1B — RESULTS COMPARISON")
print(f"{'='*len(header)}")
print(header)
print(sep)

for r in results:
    id_str = f"{r['id_pres']}" if r['id_pres'] != '-' else '-'
    print(f"| {r['exp_id']:>8} | {r['method']:>6} | {str(r['rank']):>4} | {r['clean']:>5} | {r['lr']:>7} | {r['epochs']:>2} | {r['em']:>5} | {r['cer']:>6} | {r['f05']:>5} | {id_str:>5} | {r['eval_loss']:>6} |")

print(f"{'='*len(header)}")

# Find best by EM
if results:
    best = max(results, key=lambda x: x["em"])
    print(f"\nBest: {best['exp_id']} — EM={best['em']}%, F0.5={best['f05']}, ID={best['id_pres']}%")

# Send to Telegram
tg_token = os.environ.get("TG_TOKEN", "")
tg_chat = os.environ.get("TG_CHAT_ID", "")
if tg_token and tg_chat:
    lines = ["exp030 GEC 1B Results:"]
    for r in results:
        lines.append(f"{r['exp_id']}: EM={r['em']}% F0.5={r['f05']} ID={r['id_pres']}%")
    if results:
        best = max(results, key=lambda x: x["em"])
        lines.append(f"\nBest: {best['exp_id']}")
    msg = "\n".join(lines)

    import urllib.request, urllib.parse
    data = urllib.parse.urlencode({"chat_id": tg_chat, "text": msg}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{tg_token}/sendMessage", data, timeout=10)
    except:
        pass
TABLEEOF
