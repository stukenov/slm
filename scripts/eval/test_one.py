#!/usr/bin/env python3
import urllib.request, json
data = json.dumps({"model": "test", "messages": [{"role": "user", "content": "Бүгін ауа рйы жақсы"}]}).encode()
req = urllib.request.Request("http://localhost:15128/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=30)
r = json.loads(resp.read())
print("Response:", r["choices"][0]["message"]["content"])
print("Latency:", r["latency_ms"], "ms")
