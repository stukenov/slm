# PPL Scoring Pipeline + Quality Dashboard

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the entire `sozkz-corpus-clean-v3` (13.7M texts) by perplexity using the Qwen 500M model, save scores to HF, and build an interactive Streamlit dashboard for exploring quality tiers.

**Architecture:** Two components: (1) a scoring script that runs on CloudRift GPU, streams through the dataset in batches, saves PPL per text to parquet shards, then pushes to HF; (2) a local Streamlit dashboard that loads the scored dataset and provides interactive exploration with filters, histograms, and text samples.

**Tech Stack:** PyTorch, transformers, datasets, Streamlit, Plotly, CloudRift API (REST)

---

### File Structure

| File | Purpose |
|------|---------|
| `scripts/data/ppl_score_corpus.py` | Score clean-v3 corpus using Qwen 500M, save with PPL column to HF |
| `dashboard/app.py` | Streamlit dashboard entry point, page routing |
| `dashboard/pages/__init__.py` | Package init |
| `dashboard/pages/overview.py` | PPL distribution histogram, summary stats |
| `dashboard/pages/explorer.py` | Interactive text browser with PPL/source/length filters |
| `dashboard/pages/quality_tiers.py` | Auto-tier assignment (gold/silver/bronze/reject), examples per tier |
| `dashboard/pages/export.py` | Export filtered subsets to HF |
| `dashboard/requirements.txt` | Dashboard dependencies |
| `ansible/run_ppl_score_v2.yml` | Ansible playbook to deploy and run scorer on CloudRift |

---

### Task 1: PPL Scoring Script

**Files:**
- Create: `scripts/data/ppl_score_corpus.py`

This is the core scorer. Based on the existing `scripts/data/ppl_score_dataset.py` pattern but upgraded: Qwen 500M model, saves PPL scores per row (not just filters), checkpoint/resume via parquet shards.

- [ ] **Step 1: Create the scoring script**

```python
#!/usr/bin/env python3
"""Score sozkz-corpus-clean-v3 texts by perplexity using Qwen 500M.

Computes per-text PPL, adds as column, pushes scored dataset to HF.
Saves intermediate parquet shards for crash recovery.

Usage:
    # Sample mode — score 10K texts, show distribution
    python scripts/data/ppl_score_corpus.py --sample 10000

    # Full scoring — score all 13.7M texts, push to HF
    python scripts/data/ppl_score_corpus.py --run \
        --output stukenov/sozkz-corpus-scored-kk-v1

    # Resume from last shard (after crash)
    python scripts/data/ppl_score_corpus.py --run --resume \
        --output stukenov/sozkz-corpus-scored-kk-v1
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_REPO = "stukenov/sozkz-core-qwen-500m-kk-base-v1"
TOKENIZER_REPO = "stukenov/sozkz-morphbpe-100k-kk-v1"
MAX_LENGTH = 1024
BATCH_SIZE = 64
SHARD_SIZE = 50_000  # rows per parquet shard
SHARD_DIR = "ppl_shards"


def load_model(device="cuda"):
    """Load Qwen 500M model and morphbpe-100k tokenizer."""
    logger.info("Loading tokenizer: %s", TOKENIZER_REPO)
    tok_file = hf_hub_download(TOKENIZER_REPO, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)

    logger.info("Loading model: %s", MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    model = torch.compile(model)

    # Warmup
    dummy = torch.zeros(1, 16, dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(dummy)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model loaded + compiled: %.1fM params on %s", params_m, device)
    return model, tokenizer


@torch.no_grad()
def compute_ppl_batch(texts: list[str], model, tokenizer, device, max_length=MAX_LENGTH) -> list[float]:
    """Compute per-text perplexity. Returns list of PPL values."""
    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_length, padding=True,
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    ppls = []
    for i in range(len(texts)):
        mask = attention_mask[i]
        length = mask.sum().item()
        if length <= 1:
            ppls.append(float("inf"))
            continue
        shift_logits = logits[i, :length - 1, :]
        shift_labels = input_ids[i, 1:length]
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="mean")
        ppl = math.exp(min(loss.item(), 20))
        ppls.append(ppl)
    return ppls


def get_completed_shards(shard_dir: str) -> set[int]:
    """Find which shard indices are already written."""
    p = Path(shard_dir)
    if not p.exists():
        return set()
    completed = set()
    for f in p.glob("shard_*.parquet"):
        try:
            idx = int(f.stem.split("_")[1])
            completed.add(idx)
        except (IndexError, ValueError):
            pass
    return completed


def save_shard(texts: list[str], sources: list[str], ppls: list[float], shard_idx: int, shard_dir: str):
    """Save a scored shard as parquet."""
    Path(shard_dir).mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "text": pa.array(texts, type=pa.string()),
        "source": pa.array(sources, type=pa.string()),
        "ppl": pa.array(ppls, type=pa.float32()),
    })
    out_path = Path(shard_dir) / f"shard_{shard_idx:05d}.parquet"
    pq.write_table(table, out_path)
    logger.info("Saved shard %d: %d rows -> %s", shard_idx, len(texts), out_path)


def run_sample(n_samples: int, device: str = "cuda"):
    """Score a sample and print distribution."""
    model, tokenizer = load_model(device)

    logger.info("Loading dataset (streaming)...")
    ds = load_dataset("saken-tukenov/sozkz-corpus-clean-v3", split="train", streaming=True)

    texts, sources = [], []
    for row in ds:
        if len(texts) >= n_samples:
            break
        t = row.get("text", "")
        if t and len(t.strip()) > 20:
            texts.append(t)
            sources.append(row.get("source", "unknown"))

    logger.info("Scoring %d texts...", len(texts))
    all_ppls = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        ppls = compute_ppl_batch(batch, model, tokenizer, device)
        all_ppls.extend(ppls)
        if (i // BATCH_SIZE + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = len(all_ppls) / elapsed
            logger.info("  %d/%d scored (%.0f texts/s)", len(all_ppls), len(texts), rate)

    elapsed = time.time() - t0
    ppls_arr = np.array(all_ppls)
    ppls_finite = ppls_arr[np.isfinite(ppls_arr)]

    print()
    print("=" * 70)
    print("PPL DISTRIBUTION (Qwen 500M scorer, %d texts, %.0fs)" % (len(ppls_finite), elapsed))
    print("=" * 70)
    print("  Mean:   %.1f" % np.mean(ppls_finite))
    print("  Median: %.1f" % np.median(ppls_finite))
    print("  Std:    %.1f" % np.std(ppls_finite))
    print()
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        print("  P%02d:    %.1f" % (p, np.percentile(ppls_finite, p)))

    print()
    print("PPL buckets:")
    for lo, hi in [(0, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, float("inf"))]:
        count = int(np.sum((ppls_finite >= lo) & (ppls_finite < hi)))
        pct = 100 * count / len(ppls_finite)
        hi_s = str(hi) if hi != float("inf") else "inf"
        print("  %d-%5s: %6d (%.1f%%)" % (lo, hi_s, count, pct))

    sorted_idx = np.argsort(ppls_arr)
    for label, indices in [("BEST (lowest PPL)", sorted_idx[:5]),
                           ("WORST (highest PPL)", sorted_idx[-5:]),
                           ("AROUND P90", sorted_idx[int(len(sorted_idx) * 0.89):int(len(sorted_idx) * 0.91)][:5])]:
        print()
        print("--- %s ---" % label)
        for idx in indices:
            snip = texts[idx][:200].replace("\n", " ")
            print("  [PPL=%.1f] [%s] %s" % (ppls_arr[idx], sources[idx], snip))
            print()


def run_full(output_repo: str, resume: bool = False, device: str = "cuda"):
    """Score all texts, save shards, push to HF."""
    model, tokenizer = load_model(device)

    logger.info("Loading full dataset...")
    ds = load_dataset("saken-tukenov/sozkz-corpus-clean-v3", split="train")
    total = len(ds)
    logger.info("Dataset size: %d", total)

    completed = get_completed_shards(SHARD_DIR) if resume else set()
    if completed:
        logger.info("Resuming: %d shards already done, skipping %d rows",
                     len(completed), len(completed) * SHARD_SIZE)

    shard_texts, shard_sources, shard_ppls = [], [], []
    scored = 0
    t0 = time.time()

    for i in range(0, total, BATCH_SIZE):
        current_shard = i // SHARD_SIZE
        if current_shard in completed:
            continue

        end = min(i + BATCH_SIZE, total)
        batch_rows = ds[i:end]
        batch_texts = batch_rows["text"]
        batch_sources = batch_rows["source"]

        clean_texts = [t if t and len(t.strip()) > 0 else " " for t in batch_texts]
        ppls = compute_ppl_batch(clean_texts, model, tokenizer, device)
        shard_texts.extend(batch_texts)
        shard_sources.extend(batch_sources)
        shard_ppls.extend(ppls)
        scored += len(ppls)

        if len(shard_texts) >= SHARD_SIZE:
            save_shard(shard_texts[:SHARD_SIZE], shard_sources[:SHARD_SIZE],
                       shard_ppls[:SHARD_SIZE], current_shard, SHARD_DIR)
            shard_texts = shard_texts[SHARD_SIZE:]
            shard_sources = shard_sources[SHARD_SIZE:]
            shard_ppls = shard_ppls[SHARD_SIZE:]

        if scored % 10000 < BATCH_SIZE:
            elapsed = time.time() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            eta_h = (total - scored) / rate / 3600 if rate > 0 else 0
            logger.info("  %d/%d (%.0f/s, ETA %.1fh)", scored, total, rate, eta_h)

    # Save remaining
    if shard_texts:
        final_shard = total // SHARD_SIZE
        save_shard(shard_texts, shard_sources, shard_ppls, final_shard, SHARD_DIR)

    # Merge shards and push to HF
    logger.info("Merging shards...")
    tables = []
    for f in sorted(Path(SHARD_DIR).glob("shard_*.parquet")):
        tables.append(pq.read_table(f))
    merged = pa.concat_tables(tables)
    logger.info("Merged: %d rows", merged.num_rows)

    from datasets import Dataset
    hf_ds = Dataset(merged)
    logger.info("Pushing to %s...", output_repo)
    hf_ds.push_to_hub(output_repo, private=False)
    logger.info("Done! %d texts scored and uploaded.", merged.num_rows)


def main():
    parser = argparse.ArgumentParser(description="Score corpus by PPL using Qwen 500M")
    parser.add_argument("--sample", type=int, default=0, help="Score N texts and show distribution")
    parser.add_argument("--run", action="store_true", help="Full scoring + push to HF")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved shard")
    parser.add_argument("--output", default="stukenov/sozkz-corpus-scored-kk-v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.sample > 0:
        run_sample(args.sample, args.device)
    if args.run:
        run_full(args.output, args.resume, args.device)
    if not args.sample and not args.run:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test locally with CPU on tiny sample**

Run:
```bash
cd /Users/sakentukenov/slm
python scripts/data/ppl_score_corpus.py --sample 50 --device cpu
```
Expected: prints PPL distribution + examples. Will be slow (~2-3 min) but confirms the script works.

- [ ] **Step 3: Commit**

```bash
git add scripts/data/ppl_score_corpus.py
git commit -m "feat: add PPL scorer using Qwen 500M for corpus quality scoring"
```

---

### Task 2: Ansible Playbook for CloudRift Deployment

**Files:**
- Create: `ansible/run_ppl_score_v2.yml`
- Modify: `ansible/inventory.ini` (add CloudRift host after provisioning)

This playbook deploys the scorer to a CloudRift instance and runs it in a screen session.

- [ ] **Step 1: Create the ansible playbook**

```yaml
---
- name: Run PPL scoring (Qwen 500M) on CloudRift
  hosts: cloudrift
  vars:
    project_dir: /home/riftuser/slm
    venv_path: "{{ project_dir }}/.venv"
    python: "{{ venv_path }}/bin/python3"
    script: "{{ project_dir }}/scripts/data/ppl_score_corpus.py"
    log_dir: "{{ project_dir }}/logs"
    screen_name: ppl_scoring_v2
    hf_output: stukenov/sozkz-corpus-scored-kk-v1

  tasks:
    - name: Sync scoring script
      synchronize:
        src: "{{ playbook_dir }}/../scripts/data/"
        dest: "{{ project_dir }}/scripts/data/"
        rsync_opts:
          - "--exclude=__pycache__"

    - name: Create log directory
      file:
        path: "{{ log_dir }}"
        state: directory

    - name: Install dependencies
      pip:
        name:
          - pyarrow
          - datasets
          - transformers
          - torch
          - huggingface_hub
        virtualenv: "{{ venv_path }}"

    - name: Run sample first (10K texts, sanity check)
      shell: >
        {{ python }} {{ script }} --sample 10000
        2>&1 | tee {{ log_dir }}/ppl_score_v2_sample.log
      args:
        chdir: "{{ project_dir }}"
      register: sample_result
      async: 1800
      poll: 30

    - name: Show sample results
      debug:
        msg: "{{ sample_result.stdout_lines[-40:] }}"

    - name: Launch full scoring in screen
      shell: >
        screen -dmS {{ screen_name }} bash -c '
        {{ python }} {{ script }} --run --output {{ hf_output }}
        2>&1 | tee {{ log_dir }}/ppl_score_v2_full.log'
      args:
        chdir: "{{ project_dir }}"

    - name: Verify screen session started
      shell: screen -ls | grep {{ screen_name }}
      register: screen_check

    - name: Show status
      debug:
        msg: >
          Scoring launched in screen '{{ screen_name }}'.
          Monitor: ssh riftuser@HOST "tail -f {{ log_dir }}/ppl_score_v2_full.log"
```

- [ ] **Step 2: Commit**

```bash
git add ansible/run_ppl_score_v2.yml
git commit -m "feat: add ansible playbook for PPL scoring on CloudRift"
```

---

### Task 3: Dashboard — Overview Page

**Files:**
- Create: `dashboard/requirements.txt`
- Create: `dashboard/app.py`
- Create: `dashboard/pages/__init__.py`
- Create: `dashboard/pages/overview.py`

- [ ] **Step 1: Create requirements.txt**

```
streamlit>=1.30.0
plotly>=5.18.0
datasets>=2.16.0
pyarrow>=14.0.0
pandas>=2.0.0
numpy>=1.24.0
```

- [ ] **Step 2: Create dashboard entry point**

```python
#!/usr/bin/env python3
"""SozKZ Corpus Quality Dashboard.

Interactive exploration of PPL-scored Kazakh corpus.

Usage:
    streamlit run dashboard/app.py
    streamlit run dashboard/app.py -- --local ppl_shards/
"""
import streamlit as st

st.set_page_config(
    page_title="SozKZ Corpus Quality",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_scored_dataset(source: str = "stukenov/sozkz-corpus-scored-kk-v1"):
    """Load scored dataset from HF or local parquet directory."""
    import pandas as pd
    from pathlib import Path

    if Path(source).is_dir():
        import pyarrow.parquet as pq
        import pyarrow as pa
        tables = []
        for f in sorted(Path(source).glob("*.parquet")):
            tables.append(pq.read_table(f))
        merged = pa.concat_tables(tables)
        return merged.to_pandas()
    else:
        from datasets import load_dataset
        ds = load_dataset(source, split="train")
        return ds.to_pandas()


def main():
    st.title("SozKZ Corpus Quality Dashboard")

    with st.sidebar:
        st.header("Data Source")
        source = st.text_input(
            "HF repo or local path",
            value="stukenov/sozkz-corpus-scored-kk-v1",
        )
        if st.button("Load"):
            st.session_state["source"] = source

    source = st.session_state.get("source", "stukenov/sozkz-corpus-scored-kk-v1")

    try:
        df = load_scored_dataset(source)
        st.session_state["df"] = df
    except Exception as e:
        st.error("Failed to load dataset: %s" % e)
        st.info("Run the PPL scorer first, then point to the output repo or local shards.")
        return

    page = st.sidebar.radio("Page", ["Overview", "Explorer", "Quality Tiers", "Export"])

    if page == "Overview":
        from pages.overview import render
        render(df)
    elif page == "Explorer":
        from pages.explorer import render
        render(df)
    elif page == "Quality Tiers":
        from pages.quality_tiers import render
        render(df)
    elif page == "Export":
        from pages.export import render
        render(df)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create pages/__init__.py**

Empty file — makes `pages` a Python package.

- [ ] **Step 4: Create overview page**

```python
"""Overview page: PPL distribution, summary stats, source breakdown."""
import numpy as np
import plotly.express as px
import streamlit as st


def render(df):
    st.header("Corpus Quality Overview")

    ppl = df["ppl"].values
    finite_mask = np.isfinite(ppl)
    ppl_finite = ppl[finite_mask]

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total texts", "{:,}".format(len(df)))
    col2.metric("Median PPL", "%.1f" % np.median(ppl_finite))
    col3.metric("Mean PPL", "%.1f" % np.mean(ppl_finite))
    col4.metric("Inf PPL (broken)", "{:,}".format(int((~finite_mask).sum())))

    # Percentile table
    st.subheader("Percentiles")
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    vals = [np.percentile(ppl_finite, p) for p in percentiles]
    pcol1, pcol2 = st.columns(2)
    for i, (p, v) in enumerate(zip(percentiles, vals)):
        (pcol1 if i < 4 else pcol2).metric("P%d" % p, "%.1f" % v)

    # Histogram
    st.subheader("PPL Distribution")
    max_ppl = st.slider("Max PPL for histogram", 50, 2000, 500, step=50)
    clipped = ppl_finite[ppl_finite <= max_ppl]

    fig = px.histogram(
        x=clipped, nbins=100,
        labels={"x": "Perplexity", "y": "Count"},
        title="PPL Distribution (showing {:,} / {:,} texts, PPL <= {})".format(
            len(clipped), len(ppl_finite), max_ppl),
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    # Source breakdown
    st.subheader("PPL by Source")
    fig2 = px.box(
        df[finite_mask & (df["ppl"] <= max_ppl)],
        x="source", y="ppl",
        title="PPL Distribution by Source",
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    source_stats = df[finite_mask].groupby("source")["ppl"].agg(
        ["count", "mean", "median"]
    ).reset_index().sort_values("median")
    st.dataframe(source_stats.style.format({
        "count": "{:,.0f}", "mean": "{:.1f}", "median": "{:.1f}",
    }), use_container_width=True)
```

- [ ] **Step 5: Install deps and verify dashboard launches**

```bash
cd /Users/sakentukenov/slm
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Expected: browser opens, shows "Failed to load dataset" message (no scored data yet). Confirms UI renders.

- [ ] **Step 6: Commit**

```bash
git add dashboard/requirements.txt dashboard/app.py dashboard/pages/__init__.py dashboard/pages/overview.py
git commit -m "feat: add Streamlit dashboard with overview page for corpus quality"
```

---

### Task 4: Dashboard — Explorer Page

**Files:**
- Create: `dashboard/pages/explorer.py`

- [ ] **Step 1: Create explorer page**

```python
"""Explorer page: browse texts with interactive filters."""
import numpy as np
import streamlit as st


def render(df):
    st.header("Text Explorer")

    ppl_finite = df[np.isfinite(df["ppl"])]
    ppl_min = float(ppl_finite["ppl"].min())
    ppl_max = float(min(ppl_finite["ppl"].max(), 5000))

    col1, col2, col3 = st.columns(3)

    with col1:
        ppl_range = st.slider(
            "PPL Range", ppl_min, ppl_max,
            (ppl_min, min(500.0, ppl_max)), step=1.0,
        )
    with col2:
        sources = ["All"] + sorted(df["source"].unique().tolist())
        source_filter = st.selectbox("Source", sources)
    with col3:
        min_chars = st.number_input("Min chars", 0, 50000, 0, step=100)
        max_chars = st.number_input("Max chars", 0, 500000, 500000, step=1000)

    mask = (
        (df["ppl"] >= ppl_range[0]) & (df["ppl"] <= ppl_range[1])
        & (df["text"].str.len() >= min_chars) & (df["text"].str.len() <= max_chars)
    )
    if source_filter != "All":
        mask &= df["source"] == source_filter

    filtered = df[mask]

    st.info("Showing **{:,}** / {:,} texts ({:.1f}%)".format(
        len(filtered), len(df), 100 * len(filtered) / len(df)))

    sort_by = st.radio(
        "Sort by",
        ["PPL (low to high)", "PPL (high to low)", "Length (short)", "Length (long)"],
        horizontal=True,
    )
    if sort_by == "PPL (low to high)":
        filtered = filtered.sort_values("ppl", ascending=True)
    elif sort_by == "PPL (high to low)":
        filtered = filtered.sort_values("ppl", ascending=False)
    elif sort_by == "Length (short)":
        filtered = filtered.assign(_len=filtered["text"].str.len()).sort_values("_len").drop(columns="_len")
    else:
        filtered = filtered.assign(_len=filtered["text"].str.len()).sort_values("_len", ascending=False).drop(columns="_len")

    page_size = st.selectbox("Texts per page", [10, 25, 50, 100], index=1)
    total_pages = max(1, len(filtered) // page_size)
    page_num = st.number_input("Page", 1, total_pages, 1)

    start = (page_num - 1) * page_size
    page_df = filtered.iloc[start:start + page_size]

    for _, row in page_df.iterrows():
        text = row["text"]
        char_count = len(text)
        word_count = len(text.split())
        with st.expander(
            "PPL: %.1f | %s | %s chars / %d words" % (row["ppl"], row["source"], "{:,}".format(char_count), word_count)
        ):
            st.text(text[:2000])
            if char_count > 2000:
                st.caption("... truncated (%s more chars)" % "{:,}".format(char_count - 2000))
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/pages/explorer.py
git commit -m "feat: add text explorer page with PPL/source/length filters"
```

---

### Task 5: Dashboard — Quality Tiers Page

**Files:**
- Create: `dashboard/pages/quality_tiers.py`

- [ ] **Step 1: Create quality tiers page**

```python
"""Quality Tiers page: auto-assign gold/silver/bronze/reject tiers."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

TIER_COLORS = {
    "gold": "#FFD700",
    "silver": "#C0C0C0",
    "bronze": "#CD7F32",
    "reject": "#FF4444",
}


def render(df):
    st.header("Quality Tiers")

    ppl = df["ppl"].values
    finite_mask = np.isfinite(ppl)
    ppl_finite = ppl[finite_mask]

    st.subheader("Tier Boundaries (PPL percentiles)")
    col1, col2, col3 = st.columns(3)
    with col1:
        gold_max = st.slider("Gold max (percentile)", 10, 50, 25)
    with col2:
        silver_max = st.slider("Silver max (percentile)", gold_max + 1, 70, 50)
    with col3:
        bronze_max = st.slider("Bronze max (percentile)", silver_max + 1, 95, 75)

    thresholds = {
        "gold": (0, np.percentile(ppl_finite, gold_max)),
        "silver": (np.percentile(ppl_finite, gold_max), np.percentile(ppl_finite, silver_max)),
        "bronze": (np.percentile(ppl_finite, silver_max), np.percentile(ppl_finite, bronze_max)),
        "reject": (np.percentile(ppl_finite, bronze_max), float("inf")),
    }

    def assign_tier(ppl_val):
        if not np.isfinite(ppl_val):
            return "reject"
        for tier, (lo, hi) in thresholds.items():
            if lo <= ppl_val < hi:
                return tier
        return "reject"

    df = df.copy()
    df["tier"] = df["ppl"].apply(assign_tier)

    st.subheader("Tier Summary")
    summary_rows = []
    for tier in ["gold", "silver", "bronze", "reject"]:
        tier_df = df[df["tier"] == tier]
        tier_ppl = tier_df["ppl"].values
        tier_ppl_finite = tier_ppl[np.isfinite(tier_ppl)]
        lo, hi = thresholds[tier]
        summary_rows.append({
            "Tier": tier.upper(),
            "Count": len(tier_df),
            "% of total": "%.1f%%" % (100 * len(tier_df) / len(df)),
            "PPL range": "%.0f - %.0f" % (lo, hi) if hi != float("inf") else "%.0f+" % lo,
            "Median PPL": "%.1f" % np.median(tier_ppl_finite) if len(tier_ppl_finite) > 0 else "N/A",
            "Avg chars": "%.0f" % tier_df["text"].str.len().mean() if len(tier_df) > 0 else "N/A",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    fig = px.histogram(
        df[finite_mask & (df["ppl"] <= float(np.percentile(ppl_finite, 99)))],
        x="ppl", color="tier", nbins=100,
        color_discrete_map=TIER_COLORS,
        title="PPL Distribution by Tier",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Texts")
    n_samples = st.slider("Samples per tier", 1, 10, 3)
    for tier in ["gold", "silver", "bronze", "reject"]:
        tier_df = df[df["tier"] == tier]
        if len(tier_df) == 0:
            continue
        st.markdown("### %s (%s texts)" % (tier.upper(), "{:,}".format(len(tier_df))))
        samples = tier_df.sample(min(n_samples, len(tier_df)), random_state=42)
        for _, row in samples.iterrows():
            with st.expander("PPL: %.1f | %s | %s chars" % (row["ppl"], row["source"], "{:,}".format(len(row["text"])))):
                st.text(row["text"][:1500])
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/pages/quality_tiers.py
git commit -m "feat: add quality tiers page with configurable PPL boundaries"
```

---

### Task 6: Dashboard — Export Page

**Files:**
- Create: `dashboard/pages/export.py`

- [ ] **Step 1: Create export page**

```python
"""Export page: filter and push cleaned subsets to HuggingFace."""
import io

import numpy as np
import pandas as pd
import streamlit as st


def render(df):
    st.header("Export Filtered Dataset")

    ppl = df["ppl"].values
    ppl_finite = ppl[np.isfinite(ppl)]

    st.subheader("Filter Settings")

    col1, col2 = st.columns(2)
    with col1:
        max_ppl = st.number_input(
            "Max PPL threshold",
            value=float(np.percentile(ppl_finite, 90)),
            step=10.0,
            help="Texts with PPL above this are excluded",
        )
    with col2:
        min_chars = st.number_input("Min text length (chars)", value=50, step=10)

    mask = (df["ppl"] <= max_ppl) & (df["text"].str.len() >= min_chars) & np.isfinite(df["ppl"])
    filtered = df[mask]

    st.info(
        "**{:,}** / {:,} texts pass filter ({:.1f}%) — Median PPL: {:.1f}".format(
            len(filtered), len(df), 100 * len(filtered) / len(df), filtered["ppl"].median())
    )

    st.subheader("Filtered Source Breakdown")
    source_counts = filtered.groupby("source").size().sort_values(ascending=False)
    st.dataframe(
        pd.DataFrame({
            "source": source_counts.index,
            "count": source_counts.values,
            "%": (100 * source_counts.values / len(filtered)).round(1),
        }),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Export")
    col1, col2 = st.columns(2)
    with col1:
        repo_name = st.text_input("HF repo name", value="stukenov/sozkz-corpus-gold-kk-v1")
    with col2:
        include_ppl = st.checkbox("Include PPL column in export", value=False)

    if st.button("Push to HuggingFace", type="primary"):
        with st.spinner("Pushing {:,} texts to {}...".format(len(filtered), repo_name)):
            from datasets import Dataset
            export_df = filtered if include_ppl else filtered.drop(columns=["ppl"])
            ds = Dataset.from_pandas(export_df.reset_index(drop=True))
            ds.push_to_hub(repo_name, private=False)
            st.success("Pushed {:,} texts to {}".format(len(filtered), repo_name))

    st.subheader("Download Locally")
    if st.button("Download as Parquet"):
        buf = io.BytesIO()
        filtered.to_parquet(buf, index=False)
        st.download_button(
            "Download parquet file",
            buf.getvalue(),
            file_name="sozkz_corpus_filtered.parquet",
            mime="application/octet-stream",
        )
```

- [ ] **Step 2: Commit**

```bash
git add dashboard/pages/export.py
git commit -m "feat: add export page for pushing filtered subsets to HF"
```

---

### Task 7: End-to-End Test with Mock Data

- [ ] **Step 1: Create a mock scored dataset for testing**

```bash
cd /Users/sakentukenov/slm
python3 -c "
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import random

Path('dashboard/test_data').mkdir(parents=True, exist_ok=True)

texts = [
    ('Астана - Казакстан Республикасынын астанасы. Ол елдiн солтустiк болiгiнде орналаскан.', 'wikipedia', 8.5),
    ('Бугiн ауа райы жаксы болды, коктемнiн алгашкы кундерi басталды.', 'culturax', 15.2),
    ('Казак тiлi - туркi тiлдерiнiн кыпшак тобына жатады.', 'mc4', 12.1),
    ('asdf jkl qwerty garbage text here not kazakh', 'hplt_new', 450.0),
    ('Lorem ipsum dolor sit amet consectetur adipiscing elit', 'cc100', 800.0),
] * 200

random.seed(42)
random.shuffle(texts)

table = pa.table({
    'text': [t[0] for t in texts],
    'source': [t[1] for t in texts],
    'ppl': [t[2] + random.gauss(0, t[2]*0.1) for t in texts],
})
pq.write_table(table, 'dashboard/test_data/shard_00000.parquet')
print('Created test data: %d rows' % table.num_rows)
"
```

Expected: `Created test data: 1000 rows`

- [ ] **Step 2: Launch dashboard with test data**

```bash
cd /Users/sakentukenov/slm
streamlit run dashboard/app.py
```

In the sidebar, change the data source to `dashboard/test_data` and click Load.

Verify all 4 pages:
- **Overview**: histogram shows bimodal distribution (low PPL kazakh + high PPL garbage), source box plots visible
- **Explorer**: filters work, texts display in expanders, pagination works
- **Quality Tiers**: 4 tiers with adjustable sliders, sample texts per tier
- **Export**: filter count updates, download button works

- [ ] **Step 3: Commit test data**

```bash
git add dashboard/test_data/
git commit -m "test: add mock scored data for dashboard testing"
```

---

### Task 8: Provision CloudRift and Run Scoring

This is a manual execution step — not automated code.

- [ ] **Step 1: Provision a CloudRift GPU instance**

Use the CloudRift API or dashboard to rent a single GPU instance (A100 or A10 preferred). Add it to `ansible/inventory.ini`:

```ini
[cloudrift]
ppl_scorer ansible_host=<HOST_IP> ansible_user=riftuser

[cloudrift:vars]
ansible_python_interpreter=/usr/bin/python3
ansible_ssh_common_args='-o ConnectTimeout=30 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no'
project_dir=/home/riftuser/slm
```

- [ ] **Step 2: Deploy project and run sample**

```bash
cd /Users/sakentukenov/slm
ansible-playbook ansible/deploy.yml -i ansible/inventory.ini -l cloudrift
ansible-playbook ansible/run_ppl_score_v2.yml -i ansible/inventory.ini -l cloudrift
```

Check sample results in the output. If distribution looks sane, the full run auto-launches in screen.

- [ ] **Step 3: Monitor progress**

```bash
ssh -o StrictHostKeyChecking=no riftuser@<HOST> "tail -f /home/riftuser/slm/logs/ppl_score_v2_full.log"
```

- [ ] **Step 4: Verify scored dataset on HF**

After completion, verify the pushed dataset:
```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('stukenov/sozkz-corpus-scored-kk-v1', split='train', streaming=True)
for i, row in enumerate(ds):
    if i >= 3: break
    print('PPL=%.1f [%s] %s' % (row['ppl'], row['source'], row['text'][:100]))
"
```

- [ ] **Step 5: Launch dashboard with real data**

```bash
streamlit run dashboard/app.py
```

Point to `stukenov/sozkz-corpus-scored-kk-v1` in the sidebar. Explore the real results.
