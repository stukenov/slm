#!/usr/bin/env python3
"""Benchmark vLLM GEC servers on both ports."""
import urllib.request
import json


def bench(port, text):
    data = json.dumps({"model": "test", "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(
        "http://localhost:%d/v1/chat/completions" % port,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


texts = [
    "Бүгін ауа райы жақсы",       # clean
    "Бүгін ауа рйы жақсы",        # typo
    "Мен мектепке бардым кеше",    # grammar error
]

print("=== 600M vLLM (port 15128) ===")
for t in texts:
    r = bench(15128, t)
    c = r["choices"][0]["message"]["content"]
    ms = r["latency_ms"]
    print("  [%sms] '%s' -> '%s'" % (ms, t, c))

print()
print("=== 300M vLLM (port 15129) ===")
for t in texts:
    r = bench(15129, t)
    c = r["choices"][0]["message"]["content"]
    ms = r["latency_ms"]
    tags = r.get("pipeline", {}).get("tags_triggered", [])
    print("  [%sms] '%s' -> '%s' tags=%s" % (ms, t, c, tags))

print()

# GPU memory
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("=== GPU Memory ===")
print(result.stdout)
