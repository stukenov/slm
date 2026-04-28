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
            with st.expander("PPL: %.1f | %s | %s chars" % (
                row["ppl"], row["source"], "{:,}".format(len(row["text"])))):
                st.text(row["text"][:1500])
