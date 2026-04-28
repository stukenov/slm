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
            "PPL: %.1f | %s | %s chars / %d words" % (
                row["ppl"], row["source"], "{:,}".format(char_count), word_count)
        ):
            st.text(text[:2000])
            if char_count > 2000:
                st.caption("... truncated (%s more chars)" % "{:,}".format(char_count - 2000))
