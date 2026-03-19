#!/usr/bin/env python3
"""Debug raw vLLM output to check correction logic."""
import urllib.request
import json


def raw_test(port, text):
    data = json.dumps({"model": "test", "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(
        "http://localhost:%d/v1/chat/completions" % port,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


# Check full JSON response for typo case
for port, label in [(15128, "600M"), (15129, "300M")]:
    print("=== %s (port %d) ===" % (label, port))
    r = raw_test(port, "Бүгін ауа рйы жақсы")
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print()
