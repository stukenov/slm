#!/usr/bin/env python3
"""Quick test: JSON-formatted GEC generation."""
import json, urllib.request, time, re, sys

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:15127'
seeds = open('/root/gec_seeds.txt').readlines()[:5]

EXAMPLE = '{"original": "Мен мектепке бардым.", "error": "Мен мектепте бардым.", "changed_word": "мектепке->мектепте"}'

for seed in seeds:
    seed = seed.strip()
    prompt = f'Сөйлемдегі бір сөздің септігін өзгерт. JSON форматында жауап бер.\n\nМысал:\n{EXAMPLE}\n\nСөйлем: {seed}\nJSON:'

    data = json.dumps({
        'model': 'GPT-OSS-120B',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 500,
        'temperature': 0.3,
    }).encode()
    req = urllib.request.Request(f'{URL}/v1/chat/completions', data=data, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=180)
    result = json.loads(resp.read())
    t1 = time.time()
    content = (result['choices'][0]['message'].get('content', '') or '').strip()
    toks = result.get('usage', {}).get('completion_tokens', 0)
    print(f'{t1-t0:.0f}s {toks}tok | {seed[:50]}')
    print(f'  -> {content[:150]}')
    try:
        m = re.search(r'\{[^}]+\}', content)
        if m:
            j = json.loads(m.group())
            print(f'  PARSED: err={j.get("error","?")[:60]}, changed={j.get("changed_word","?")}')
        else:
            print(f'  NO JSON found')
    except Exception as e:
        print(f'  JSON parse error: {e}')
    print()
