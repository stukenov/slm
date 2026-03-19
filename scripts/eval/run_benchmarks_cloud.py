#!/usr/bin/env python3
"""Run both benchmarks on a remote GPU instance and collect results.

Usage:
    PYTHONPATH=src .venv-cloud/bin/python scripts/run_benchmarks_cloud.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

MODEL_ID = "stukenov/sozkz-core-llama-150m-kk-instruct-v2"

# Read secrets
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not OPENAI_API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), "..", "Kaz-Offline-Arena", ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("OPENAI_API_KEY="):
                OPENAI_API_KEY = line.strip().split("=", 1)[1]
            elif line.startswith("HUGGINGFACE_TOKEN="):
                HF_TOKEN = HF_TOKEN or line.strip().split("=", 1)[1]

if not HF_TOKEN:
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        HF_TOKEN = open(token_path).read().strip()


def build_remote_script():
    return f'''#!/bin/bash
set -e

export HF_TOKEN="{HF_TOKEN}"
export OPENAI_API_KEY="{OPENAI_API_KEY}"

echo "=== Installing dependencies ==="
pip install -q torch transformers datasets tqdm openai pydantic tenacity python-dotenv huggingface_hub 2>&1 | tail -3

echo "=== Downloading model ==="
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('{MODEL_ID}')
AutoModelForCausalLM.from_pretrained('{MODEL_ID}')
print('Model cached.')
"

echo "=== Running MC Bench ==="
python3 << 'MC_SCRIPT'
import json, torch
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

CHOICES = ["A", "B", "C", "D"]
model_id = "{MODEL_ID}"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

vocab = set(tokenizer.get_vocab().keys())
fmt = "chatml" if "<|user|>" in vocab else "alpaca"
print(f"Format: {{fmt}}")

choice_ids = dict()
for c in CHOICES:
    choice_ids[c] = tokenizer.encode(c, add_special_tokens=False)[0]

ds = load_dataset("kz-transformers/kk-socio-cultural-bench-mc", split="train")
correct, total = 0, 0
by_cat = defaultdict(lambda: dict(correct=0, total=0))

for row in tqdm(ds, desc="MC"):
    options = chr(10).join(f"{{c}}) {{row[c]}}" for c in CHOICES)
    if fmt == "chatml":
        prompt = "<|user|>" + chr(10) + row["question"] + chr(10) + chr(10) + options + chr(10) + chr(10) + "Дұрыс жауапты таңдаңыз. Тек A, B, C немесе D әрпін жазыңыз." + chr(10) + "<|end|>" + chr(10) + "<|assistant|>" + chr(10)
    else:
        prompt = "### Нұсқаулық:" + chr(10) + row["question"] + chr(10) + chr(10) + options + chr(10) + chr(10) + "### Жауап:" + chr(10)
    ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    scores = dict()
    for c in CHOICES:
        scores[c] = logits[choice_ids[c]].item()
    pred = max(scores, key=scores.get)
    gold = row["answer"]
    cat = row["category"]
    hit = pred == gold
    correct += hit
    total += 1
    by_cat[cat]["total"] += 1
    by_cat[cat]["correct"] += hit

acc = correct / total if total else 0
print(f"Overall: {{correct}}/{{total}} = {{acc:.4f}} ({{acc*100:.1f}}%)")
cat_results = dict()
for cat in sorted(by_cat):
    c2, t2 = by_cat[cat]["correct"], by_cat[cat]["total"]
    a2 = c2/t2 if t2 else 0
    cat_results[cat] = dict(correct=c2, total=t2, accuracy=round(a2, 4))
    print(f"  {{cat}}: {{c2}}/{{t2}} = {{a2*100:.1f}}%")

results = dict(model=model_id, total=total, correct=correct, accuracy=round(acc, 4), categories=cat_results)
with open("/tmp/mc_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
MC_SCRIPT

echo "=== Running Arena Inference ==="
python3 << 'ARENA_SCRIPT'
import json, uuid, torch, math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_id = "{MODEL_ID}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset("kz-transformers/arena-offline-qa", split="arena")
df = dataset.to_pandas()

qtypes = ["WHY_QS", "WHAT_QS", "HOW_QS", "DESCRIBE_QS", "ANALYZE_QS"]
n = len(qtypes)
df["selected_qt"] = [qtypes[i % n] for i in range(len(df))]

system_msg = "Сен білімді көмекшісің. Контекстке сүйеніп, сұраққа қазақ тілінде жауап бер."

MAX_PROMPT_TOKENS = 720
MAX_NEW_TOKENS = 300

outputs = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Arena"):
    qt = row["selected_qt"]
    qtext = row[qt]
    if not qtext or str(qtext) == "nan":
        continue
    context_text = row["text"]
    question_text = f"Сұрақ: {{qtext}}"
    suffix = chr(10) + chr(10) + "Контекстке сүйеніп, қазақша қысқа және нақты жауап бер."
    template_pre = "<|system|>" + chr(10) + system_msg + chr(10) + "<|end|>" + chr(10) + "<|user|>" + chr(10)
    template_post = chr(10) + "<|end|>" + chr(10) + "<|assistant|>" + chr(10)
    overhead_ids = tokenizer.encode(template_pre + question_text + suffix + template_post, add_special_tokens=False)
    overhead_len = len(overhead_ids)
    max_ctx_tokens = MAX_PROMPT_TOKENS - overhead_len
    if max_ctx_tokens < 50:
        max_ctx_tokens = 50
    ctx_ids = tokenizer.encode(context_text, add_special_tokens=False)
    if len(ctx_ids) > max_ctx_tokens:
        ctx_ids = ctx_ids[:max_ctx_tokens]
        context_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)
    prompt_text = context_text + chr(10) + question_text + suffix
    chatml = template_pre + prompt_text + template_post
    input_ids = tokenizer.encode(chatml, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.4,
            do_sample=True,
            top_p=0.92,
            top_k=40,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer_ids = gen_ids[0][input_ids.shape[1]:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    for stop in ["<|end|>", "<|user|>", "<|system|>"]:
        if stop in answer:
            answer = answer[:answer.index(stop)].strip()
    outputs.append(dict(
        task_id=idx, question_type=qt, context=row["text"],
        prompt=prompt_text, output=answer,
        tokens_count=answer_ids.size(0),
        model=model_id, generation_id=str(uuid.uuid4()),
    ))

tc = [r["tokens_count"] for r in outputs]
avg_tok = sum(tc) / len(tc) if tc else 0

result = dict(summary=dict(avg_tokens=avg_tok, model_type="SFT"), results=outputs)
with open("/tmp/arena_inference.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"Saved {{len(outputs)}} items, avg_tokens={{avg_tok:.1f}}")
ARENA_SCRIPT

echo "=== Running Arena Judge ==="
python3 << 'JUDGE_SCRIPT'
import json, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def call_judge(prompt):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            dict(role="system", content=(
                "You are an expert evaluator with deep knowledge of Kazakh language, culture, history and context of Kazakhstan. "
                "For the given context, question, and model response, evaluate the quality of the response. Provide an explanation "
                "and assign a score between 0 and 10. 0 is completely irrelevant and unhelpful, 3 is partially relevant and helpful but incorrect, "
                "5 is somewhat relevant and helpful but not fully correct, 7 is mostly relevant and helpful with some issues, "
                "10 is completely relevant and helpful with no issues. Return a JSON object with keys 'explanation' and 'score'."
            )),
            dict(role="user", content=prompt),
        ],
        max_tokens=3000,
    )
    return resp.choices[0].message.content

def clean_json(s):
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    return m.group(1) if m else s.strip()

with open("/tmp/arena_inference.json") as f:
    data = json.load(f)

records = data["results"]
print(f"Judging {{len(records)}} records...")

def judge_one(rec):
    prompt = f"**Context**: {{rec['context']}}" + chr(10)*2 + f"**Question**: ({{rec['question_type']}})" + chr(10)*2 + f"**Response**: {{rec['output']}}" + chr(10)*2 + "Evaluate the response and return JSON with 'explanation' and 'score' (0-10)."
    try:
        resp = call_judge(prompt)
        obj = json.loads(clean_json(resp))
        return dict(task_id=rec["task_id"], question_type=rec["question_type"], score=obj["score"], success=True)
    except Exception as e:
        return dict(task_id=rec["task_id"], question_type=rec["question_type"], score=0, success=False, error=str(e))

judges = []
with ThreadPoolExecutor(max_workers=50) as ex:
    futs = [ex.submit(judge_one, r) for r in records]
    for i, f in enumerate(as_completed(futs), 1):
        judges.append(f.result())
        if i % 50 == 0:
            print(f"  {{i}}/{{len(records)}}")

scores = [j["score"] for j in judges if j["success"]]
avg_score = sum(scores) / len(scores) if scores else 0

by_type = dict()
for j in judges:
    qt = j["question_type"]
    if qt not in by_type:
        by_type[qt] = []
    if j["success"]:
        by_type[qt].append(j["score"])

print(f"Overall: {{avg_score:.2f}} / 10")
print(f"Avg tokens: {{data['summary']['avg_tokens']:.1f}}")
for qt in sorted(by_type):
    s = by_type[qt]
    print(f"  {{qt}}: {{sum(s)/len(s):.2f}}")

result = dict(
    model=records[0]["model"] if records else "",
    avg_score=round(avg_score, 2),
    avg_tokens=round(data["summary"]["avg_tokens"], 1),
    by_type=dict((qt, round(sum(s)/len(s), 2)) for qt, s in by_type.items()),
    judges=judges
)
with open("/tmp/arena_results.json", "w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
JUDGE_SCRIPT

echo "=== ALL DONE ==="
cat /tmp/mc_results.json
echo "---ARENA---"
python3 -c "
import json
with open('/tmp/arena_results.json') as f:
    d = json.load(f)
print(f'Arena: ' + str(d['avg_score']) + '/10, avg_tokens=' + str(d['avg_tokens']))
for qt, s in d['by_type'].items():
    print(f'  ' + qt + ': ' + str(s))
"
'''


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from slm.cloud import vastai

    print("Searching for cheap GPU...")
    offers = vastai.search_offers(
        "rentable=true num_gpus=1 compute_cap>=800 compute_cap<=890 gpu_ram>=20 "
        "dph<=0.60 inet_down>=500 disk_space>=30"
    )
    if not offers:
        offers = vastai.search_offers(
            "rentable=true num_gpus=1 compute_cap>=750 compute_cap<=890 gpu_ram>=16 "
            "dph<=0.50 inet_down>=200 disk_space>=30"
        )
    if not offers:
        print("ERROR: No GPU offers")
        sys.exit(1)

    best = sorted(offers, key=lambda x: float(x.get("dph_total", 99)))[0]
    offer_id = int(best["id"])
    gpu = best.get("gpu_name", "?")
    price = float(best.get("dph_total", 0))
    print(f"Selected: {gpu} @ ${price:.3f}/hr (offer {offer_id})")

    print("Creating instance...")
    instance_id = vastai.create_instance(
        offer_id,
        image="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
        disk=30,
        label="slm-benchmarks",
    )
    print(f"Instance: {instance_id}")

    try:
        print("Waiting for instance...")
        vastai.wait_for_instance(instance_id, timeout=600)
        host, port = vastai.ssh_url(instance_id)
        print(f"SSH: {host}:{port}")

        for attempt in range(30):
            try:
                r = subprocess.run(
                    ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                     "-o", "BatchMode=yes", "-p", str(port), f"root@{host}", "echo ok"],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    print(f"SSH ready (attempt {attempt+1})")
                    break
            except subprocess.TimeoutExpired:
                pass
            time.sleep(10)
        else:
            raise RuntimeError("SSH not reachable")

        script_content = build_remote_script()
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port),
             script_path, f"root@{host}:/tmp/bench.sh"],
            check=True, timeout=30,
        )
        os.unlink(script_path)

        print("Running benchmarks (est. 15-30 min)...")
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=60",
             "-p", str(port), f"root@{host}",
             "bash /tmp/bench.sh"],
            capture_output=True, text=True, timeout=7200,
        )
        output = result.stdout
        print(output[-5000:] if len(output) > 5000 else output)
        if result.returncode != 0 and result.stderr:
            print("STDERR:", result.stderr[-2000:])

        os.makedirs("results", exist_ok=True)
        for remote, local in [
            ("/tmp/mc_results.json", "results/eval_mc_bench_v2.json"),
            ("/tmp/arena_inference.json", "results/arena_inference_v2.json"),
            ("/tmp/arena_results.json", "results/arena_results_v2.json"),
        ]:
            subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(port),
                 f"root@{host}:{remote}", local],
                timeout=30,
            )
            print(f"Downloaded {local}")

    finally:
        print(f"Destroying instance {instance_id}...")
        vastai.destroy_instance(instance_id)
        print("Done.")


if __name__ == "__main__":
    main()
