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
