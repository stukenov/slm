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
    buf = io.BytesIO()
    filtered.to_parquet(buf, index=False)
    st.download_button(
        "Download as Parquet",
        buf.getvalue(),
        file_name="sozkz_corpus_filtered.parquet",
        mime="application/octet-stream",
    )
