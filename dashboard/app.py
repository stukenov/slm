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
def load_scored_dataset(source):
    """Load scored dataset from HF or local parquet directory."""
    from pathlib import Path

    if Path(source).is_dir():
        import pyarrow as pa
        import pyarrow.parquet as pq
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
