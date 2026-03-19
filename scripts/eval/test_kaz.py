import json, urllib.request, time

def ask(prompt, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    data = json.dumps({
        "model": "gpt-oss-120b",
        "messages": msgs,
        "max_tokens": 500,
        "temperature": 0.7
    }).encode()
    req = urllib.request.Request("http://localhost:8080/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=180)
    result = json.loads(resp.read())
    elapsed = time.time() - t0
    ch = result["choices"][0]["message"]
    tg = result["timings"]["predicted_per_second"]
    pp = result["timings"]["prompt_per_second"]
    print(f"\n{'='*60}")
    print(f"Q: {prompt[:80]}")
    print(f"TG: {tg:.1f} t/s | PP: {pp:.1f} t/s | Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    if ch.get("reasoning_content"):
        print(f"[Reasoning]: {ch['reasoning_content'][:300]}")
    print(f"\n{ch['content']}\n")

sys_kk = "You are a helpful assistant that ALWAYS responds in Kazakh language (qazaq tili). Use Kazakh script."

ask("Qazaqstan turaly 5 soilem zhaz.", sys_kk)
ask("Abai Qunanbaev kim? Onyn shygarmashylygy turaly aityp ber.", sys_kk)
ask("Qazaq tilindegi en tanimal maqal-matelderdyn 5-euin zhaz zhane olardy tusindirip ber.", sys_kk)
ask("Qazirgi Qazaqstannyn ekonomikasy qalai damyp kele zhatyr?", sys_kk)
