"""Demo: show each cleaning stage on random real texts."""
import pyarrow.parquet as pq
import random, os, re, unicodedata, gzip
from collections import Counter

random.seed(42)
samples = []

for d in ['/root/slm/data/collected', '/root/slm/data/collected_wave2']:
    for f in os.listdir(d):
        if not f.endswith('.parquet'):
            continue
        path = os.path.join(d, f)
        try:
            t = pq.read_table(path, columns=['text'])
            n = len(t)
            idxs = random.sample(range(n), min(3, n))
            for i in idxs:
                txt = t.column('text')[i].as_py()
                samples.append((f.replace('.parquet', ''), txt))
        except Exception:
            pass

random.shuffle(samples)
samples = samples[:8]

RE_CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
RE_MULTI_SPACE = re.compile(r'[ \t]+')
RE_MULTI_NEWLINE = re.compile(r'\n{3,}')
RE_URL = re.compile(r'https?://\S+', re.IGNORECASE)
RE_HTML_TAG = re.compile(r'<[^>]{1,200}>')
RE_KAZAKH_CHAR = re.compile(r'[\u04D8\u04D9\u0492\u0493\u049A\u049B\u04A2\u04A3\u04E8\u04E9\u04B0\u04B1\u04AE\u04AF\u04BA\u04BB\u0406\u0456]')


def preview(text, max_len=250):
    p = text[:max_len].replace('\n', '\\n')
    if len(text) > max_len:
        p += '...'
    return p


for idx, (source, raw) in enumerate(samples):
    print()
    print('=' * 70)
    print(f'SAMPLE {idx+1} (source: {source}, raw len: {len(raw)})')
    print('=' * 70)
    print(f'  RAW: {preview(raw)}')

    text = raw

    # 1. NFC
    text_nfc = unicodedata.normalize('NFC', text)
    if text_nfc != text:
        print(f'  [1 NFC] CHANGED (diff {len(text) - len(text_nfc)} chars)')
    else:
        print(f'  [1 NFC] no change')
    text = text_nfc

    # 2. Control chars
    text2 = RE_CONTROL.sub('', text)
    removed = len(text) - len(text2)
    if removed:
        print(f'  [2 CONTROL] removed {removed} control chars')
    else:
        print(f'  [2 CONTROL] no change')
    text = text2

    # 3. Whitespace
    text3 = RE_MULTI_SPACE.sub(' ', text)
    text3 = RE_MULTI_NEWLINE.sub('\n\n', text3)
    text3 = text3.strip()
    if text3 != text.strip():
        print(f'  [3 WHITESPACE] {len(text)} -> {len(text3)} chars')
        print(f'    AFTER: {preview(text3)}')
    else:
        print(f'  [3 WHITESPACE] no change')
    text = text3

    # 4. Min length
    if len(text) < 50:
        print(f'  [4 MIN_LEN] REJECTED (len={len(text)} < 50)')
        print()
        continue
    else:
        print(f'  [4 MIN_LEN] OK (len={len(text)})')

    # 5. Kazakh chars
    has_kaz = bool(RE_KAZAKH_CHAR.search(text))
    if not has_kaz:
        print(f'  [5 KAZ_CHAR] REJECTED - no kazakh-specific chars')
        print()
        continue
    else:
        # show which chars found
        found = set(RE_KAZAKH_CHAR.findall(text))
        print(f'  [5 KAZ_CHAR] OK - found: {" ".join(sorted(found))}')

    # 6. URL density
    urls = RE_URL.findall(text)
    text_len = len(text) or 1
    density = len(urls) / (text_len / 1000)
    if density > 5:
        print(f'  [6 URL] REJECTED - {len(urls)} URLs, density={density:.1f}')
        print()
        continue
    else:
        print(f'  [6 URL] OK ({len(urls)} URLs, density={density:.1f})')

    # 7. HTML
    html_count = len(RE_HTML_TAG.findall(text))
    if html_count > 5:
        print(f'  [7 HTML] REJECTED - {html_count} tags')
        print()
        continue
    else:
        print(f'  [7 HTML] OK ({html_count} tags)')

    # 8. Script profile
    counts = Counter()
    for ch in text:
        if ch.isspace():
            continue
        if '\u0400' <= ch <= '\u04ff' or '\u0500' <= ch <= '\u052f':
            counts['cyr'] += 1
        elif ch.isascii() and ch.isalpha():
            counts['lat'] += 1
        elif ch.isdigit():
            counts['dig'] += 1
        else:
            counts['oth'] += 1
    total = sum(counts.values()) or 1
    print(f'  [8 SCRIPT] cyr={counts["cyr"]/total*100:.0f}% lat={counts["lat"]/total*100:.0f}% dig={counts["dig"]/total*100:.0f}% oth={counts["oth"]/total*100:.0f}%')

    # 9. Gzip ratio
    encoded = text.encode('utf-8')
    compressed = gzip.compress(encoded, compresslevel=6)
    ratio = len(compressed) / len(encoded) if encoded else 1.0
    status = 'SUSPICIOUS' if ratio < 0.3 else 'OK'
    print(f'  [9 GZIP] ratio={ratio:.3f} ({status})')

    # Final
    print(f'  CLEAN: {preview(text)}')
    print()

print('Done.')
