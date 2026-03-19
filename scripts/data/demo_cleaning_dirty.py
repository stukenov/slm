"""Find real examples where cleaning REJECTS or CHANGES text."""
import pyarrow.parquet as pq
import random, os, re, unicodedata, gzip, sys
from collections import Counter

random.seed(123)

RE_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
RE_MULTI_SPACE = re.compile(r'[ \t]+')
RE_MULTI_NEWLINE = re.compile(r'\n{3,}')
RE_URL = re.compile(r'https?://\S+', re.IGNORECASE)
RE_HTML_TAG = re.compile(r'<[^>]{1,200}>')
RE_KAZAKH_CHAR = re.compile(r'[\u04D8\u04D9\u0492\u0493\u049A\u049B\u04A2\u04A3\u04E8\u04E9\u04B0\u04B1\u04AE\u04AF\u04BA\u04BB\u0406\u0456]')

# Load fasttext
import numpy as _np
# Patch numpy 2.x before importing fasttext
_orig_array = _np.array
def _safe_array(*a, **kw):
    kw.pop('copy', None)
    return _orig_array(*a, **kw)
_np.array = _safe_array

import fasttext
fasttext.FastText.eprint = lambda x: None
ft_model = fasttext.load_model('/root/slm/models/lid.176.bin')

def ft_predict(text):
    line = text.replace('\n', ' ')[:5000]
    result = ft_model.predict(line, k=3)
    labels = result[0]
    scores = list(result[1]) if hasattr(result[1], '__iter__') else [result[1]]
    return {l.replace('__label__', ''): float(s) for l, s in zip(labels, scores)}

def preview(text, max_len=200):
    p = text[:max_len].replace('\n', '\\n')
    if len(text) > max_len:
        p += '...'
    return p

# Collect all texts with source
all_texts = []
for d in ['/root/slm/data/collected', '/root/slm/data/collected_wave2']:
    for f in os.listdir(d):
        if not f.endswith('.parquet'):
            continue
        path = os.path.join(d, f)
        try:
            t = pq.read_table(path, columns=['text'])
            n = len(t)
            idxs = random.sample(range(n), min(500, n))
            for i in idxs:
                txt = t.column('text')[i].as_py()
                all_texts.append((f.replace('.parquet', ''), txt))
        except Exception:
            pass

random.shuffle(all_texts)
print(f'Loaded {len(all_texts)} random samples\n')

# Categories of interesting finds
found = {
    'control_chars': [],
    'whitespace_changed': [],
    'too_short': [],
    'no_kazakh_chars': [],
    'high_url': [],
    'html_heavy': [],
    'not_kazakh_lid': [],
    'low_gzip': [],
    'high_latin': [],
    'mixed_lang': [],
}

for source, raw in all_texts:
    text = raw
    entry = (source, raw)

    # 1. NFC
    text = unicodedata.normalize('NFC', text)

    # 2. Control chars
    text2 = RE_CONTROL.sub('', text)
    if len(text2) < len(text) and len(found['control_chars']) < 2:
        found['control_chars'].append(entry)
    text = text2

    # 3. Whitespace
    text3 = RE_MULTI_SPACE.sub(' ', text)
    text3 = RE_MULTI_NEWLINE.sub('\n\n', text3)
    text3 = text3.strip()
    if text3 != text.strip() and len(found['whitespace_changed']) < 2:
        found['whitespace_changed'].append((source, raw, text3))
    text = text3

    # 4. Too short
    if len(text) < 50:
        if len(found['too_short']) < 2:
            found['too_short'].append(entry)
        continue

    # 5. No kazakh chars
    if not RE_KAZAKH_CHAR.search(text):
        if len(found['no_kazakh_chars']) < 2:
            found['no_kazakh_chars'].append(entry)
        continue

    # 6. URL density
    urls = RE_URL.findall(text)
    density = len(urls) / (len(text) / 1000) if text else 0
    if density > 5 and len(found['high_url']) < 2:
        found['high_url'].append(entry)

    # 7. HTML
    html_count = len(RE_HTML_TAG.findall(text))
    if html_count > 5 and len(found['html_heavy']) < 2:
        found['html_heavy'].append(entry)

    # 8. Script
    counts = Counter()
    for ch in text:
        if ch.isspace(): continue
        if '\u0400' <= ch <= '\u04ff' or '\u0500' <= ch <= '\u052f': counts['cyr'] += 1
        elif ch.isascii() and ch.isalpha(): counts['lat'] += 1
        elif ch.isdigit(): counts['dig'] += 1
        else: counts['oth'] += 1
    total = sum(counts.values()) or 1
    lat_pct = counts['lat'] / total
    if lat_pct > 0.3 and len(found['high_latin']) < 2:
        found['high_latin'].append(entry)

    # 9. FastText LID
    lid = ft_predict(text)
    kk_score = lid.get('kk', 0)
    top_lang = max(lid, key=lid.get)
    if kk_score < 0.5 and len(found['not_kazakh_lid']) < 2:
        found['not_kazakh_lid'].append((source, raw, lid))
    elif top_lang == 'kk' and kk_score < 0.8 and len(found['mixed_lang']) < 2:
        found['mixed_lang'].append((source, raw, lid))

    # 10. Gzip
    encoded = text.encode('utf-8')
    compressed = gzip.compress(encoded, compresslevel=6)
    ratio = len(compressed) / len(encoded) if encoded else 1.0
    if ratio < 0.25 and len(found['low_gzip']) < 2:
        found['low_gzip'].append((source, raw, ratio))


# Print results
print('=' * 70)
print('REJECTED: TOO SHORT (<50 chars)')
print('=' * 70)
for src, txt in found['too_short']:
    print(f'  [{src}] "{preview(txt)}"')
print()

print('=' * 70)
print('REJECTED: NO KAZAKH-SPECIFIC CHARS (missing \u04d9,\u0493,\u049b,\u04a3,\u04e9,\u04b1,\u04af...)')
print('=' * 70)
for src, txt in found['no_kazakh_chars']:
    print(f'  [{src}] "{preview(txt)}"')
print()

print('=' * 70)
print('MODIFIED: CONTROL CHARS REMOVED')
print('=' * 70)
for src, txt in found['control_chars']:
    ctrl = [(i, repr(ch)) for i, ch in enumerate(txt[:500]) if RE_CONTROL.match(ch)]
    print(f'  [{src}] found control chars at positions: {ctrl[:10]}')
    print(f'    TEXT: "{preview(txt)}"')
print()

print('=' * 70)
print('MODIFIED: WHITESPACE COLLAPSED')
print('=' * 70)
for item in found['whitespace_changed']:
    src, raw, cleaned = item
    print(f'  [{src}] {len(raw)} -> {len(cleaned)} chars')
    print(f'    BEFORE: "{preview(raw)}"')
    print(f'    AFTER:  "{preview(cleaned)}"')
print()

print('=' * 70)
print('REJECTED: HIGH URL DENSITY (>5 per 1K chars)')
print('=' * 70)
for src, txt in found['high_url']:
    urls = RE_URL.findall(txt)
    print(f'  [{src}] {len(urls)} URLs in {len(txt)} chars')
    print(f'    TEXT: "{preview(txt)}"')
print()

print('=' * 70)
print('REJECTED: TOO MUCH HTML (>5 tags)')
print('=' * 70)
for src, txt in found['html_heavy']:
    tags = RE_HTML_TAG.findall(txt)
    print(f'  [{src}] {len(tags)} HTML tags')
    print(f'    TEXT: "{preview(txt)}"')
    print(f'    TAGS: {tags[:8]}')
print()

print('=' * 70)
print('REJECTED: HIGH LATIN % (>30%)')
print('=' * 70)
for src, txt in found['high_latin']:
    print(f'  [{src}] "{preview(txt)}"')
print()

print('=' * 70)
print('REJECTED by FastText LID: NOT KAZAKH (kk < 0.5)')
print('=' * 70)
for item in found['not_kazakh_lid']:
    src, txt, lid = item
    print(f'  [{src}] LID={lid}')
    print(f'    TEXT: "{preview(txt)}"')
print()

print('=' * 70)
print('WARNING: MIXED LANGUAGE (kk=0.5-0.8, uncertain)')
print('=' * 70)
for item in found['mixed_lang']:
    src, txt, lid = item
    print(f'  [{src}] LID={lid}')
    print(f'    TEXT: "{preview(txt)}"')
print()

print('=' * 70)
print('SUSPICIOUS: LOW GZIP RATIO (<0.25 = very repetitive)')
print('=' * 70)
for item in found['low_gzip']:
    src, txt, ratio = item
    print(f'  [{src}] gzip={ratio:.3f}')
    print(f'    TEXT: "{preview(txt, 300)}"')
print()

# Summary
print('=' * 70)
print('SUMMARY: how many found in sample')
print('=' * 70)
for cat, items in found.items():
    print(f'  {cat}: {len(items)} examples found')
