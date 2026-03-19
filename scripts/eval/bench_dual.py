#!/usr/bin/env python3
"""Benchmark: 2 instances simultaneously vs 1 instance"""
import json, urllib.request, time, threading

PROMPT = "Write a detailed essay about the history of Central Asia, covering the Silk Road, nomadic empires, the Soviet era, and modern independence. Include at least 500 words."

def query(port, label):
    data = json.dumps({
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 300,
        "temperature": 0.7
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    elapsed = time.time() - t0
    tg = result["timings"]["predicted_per_second"]
    pp = result["timings"]["prompt_per_second"]
    n_tok = result["timings"]["predicted_n"]
    print(f"  [{label}] TG: {tg:.1f} t/s | PP: {pp:.1f} t/s | {n_tok} tokens in {elapsed:.1f}s")
    return {"tg": tg, "pp": pp, "tokens": n_tok, "elapsed": elapsed}

print("=" * 60)
print("TEST 1: Sequential - one request at a time")
print("=" * 60)

print("\n  Port 15127 (GPU 0):")
r1 = query(15127, "GPU0")
print("  Port 15128 (GPU 1):")
r2 = query(15128, "GPU1")
print(f"\n  Sequential total: {r1['elapsed'] + r2['elapsed']:.1f}s")
print(f"  Avg TG: {(r1['tg'] + r2['tg'])/2:.1f} t/s")

print("\n" + "=" * 60)
print("TEST 2: Parallel - both GPUs at same time")
print("=" * 60)

results = {}
def run_query(port, label):
    results[label] = query(port, label)

t0 = time.time()
t1 = threading.Thread(target=run_query, args=(15127, "GPU0"))
t2 = threading.Thread(target=run_query, args=(15128, "GPU1"))
t1.start()
t2.start()
t1.join()
t2.join()
wall_time = time.time() - t0

tg0 = results["GPU0"]["tg"]
tg1 = results["GPU1"]["tg"]
tok0 = results["GPU0"]["tokens"]
tok1 = results["GPU1"]["tokens"]

print(f"\n  Parallel wall time: {wall_time:.1f}s")
print(f"  GPU0 TG: {tg0:.1f} t/s | GPU1 TG: {tg1:.1f} t/s")
print(f"  Combined throughput: {(tok0 + tok1) / wall_time:.1f} effective t/s")
print(f"  Total tokens: {tok0 + tok1}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  Single instance (best):  42.4 t/s (1 client)")
print(f"  Dual sequential avg:     {(r1['tg'] + r2['tg'])/2:.1f} t/s per client")
print(f"  Dual parallel combined:  {(tok0 + tok1) / wall_time:.1f} t/s total throughput")
print(f"  Dual parallel per-GPU:   {tg0:.1f} / {tg1:.1f} t/s")
