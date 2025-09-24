import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from app_utils import load_dataset, basic_stats, detect_outliers, zscore_outliers, summarize_text_counts

st.set_page_config(page_title="Financial Transactions Dashboard", layout="wide")

@st.cache_data(show_spinner=False)
def _load(path: str):
    return load_dataset(path)

def pick_default_path():
    # Try local filename first, then fallback to the known mounted path
    candidates = [
        "financial_transactions.csv",
        "/mnt/data/financial_transactions.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""

# Header
st.title("💹 Financial Transactions – Interactive Dashboard")

# Data source controls
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    default_path = pick_default_path()
    st.caption("If you don't upload a file, I'll try to load:")
    st.code(default_path or "(no default found)")
    run_demo = st.toggle("Use bundled demo (auto-load default)", value=True if default_path else False)

if uploaded is not None:
    df = load_dataset(uploaded)
elif run_demo and default_path:
    df = _load(default_path)
else:
    st.info("Please upload a CSV file with columns like: date, amount, category, merchant, payment_method, account_type, transaction_type, description.")
    st.stop()

stats = basic_stats(df)
numeric_cols = stats["numeric_cols"]
cat_cols = [c for c in stats["categorical_cols"] if c not in ["date", "timestamp", "datetime"]]

# Column remapping for common names
date_col = None
for c in df.columns:
    cl = c.lower()
    if "date" in cl or "time" in cl:
        date_col = c
        break

amount_col_guess = None
for c in numeric_cols:
    if "amount" in c.lower() or "value" in c.lower() or "amt" in c.lower():
        amount_col_guess = c
        break

with st.sidebar:
    st.subheader("🔎 Filters")
    # Date filter
    if date_col and np.issubdtype(df[date_col].dtype, np.datetime64):
        min_d = pd.to_datetime(df[date_col]).min()
        max_d = pd.to_datetime(df[date_col]).max()
        start, end = st.date_input("Date range", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
        if isinstance(start, tuple):
            start, end = start  # streamlit older versions quirk
        df = df[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end) + pd.Timedelta(days=1))]
    else:
        st.caption("No parsed date column found; date filters disabled.")

    # Amount range
    if amount_col_guess:
        min_amt = float(np.nanmin(df[amount_col_guess]))
        max_amt = float(np.nanmax(df[amount_col_guess]))
        lo, hi = st.slider("Amount range", min_amt, max_amt, (min_amt, max_amt))
        df = df[(df[amount_col_guess] >= lo) & (df[amount_col_guess] <= hi)]

    # Category-like filters
    multi_filters = {}
    for c in ["category", "merchant", "payment_method", "account_type", "transaction_type"]:
        if c in df.columns:
            vals = sorted([v for v in df[c].dropna().astype(str).unique()])
            chosen = st.multiselect(f"{c.replace('_',' ').title()}", vals)
            if chosen:
                df = df[df[c].astype(str).isin(chosen)]

# KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Transactions", f"{len(df):,}")
with kpi2:
    total = df[amount_col_guess].sum() if amount_col_guess else float("nan")
    st.metric("Total Amount", f"{total:,.2f}" if total==total else "—")
with kpi3:
    avg = df[amount_col_guess].mean() if amount_col_guess else float("nan")
    st.metric("Avg Amount", f"{avg:,.2f}" if avg==avg else "—")
with kpi4:
    cats = df["category"].nunique() if "category" in df.columns else 0
    st.metric("Categories", f"{cats:,}")

st.markdown("---")

# Time series
if date_col and amount_col_guess and np.issubdtype(df[date_col].dtype, np.datetime64):
    ts = df.groupby(pd.Grouper(key=date_col, freq="D"))[amount_col_guess].sum().reset_index()
    fig = px.line(ts, x=date_col, y=amount_col_guess, title="Daily Spend/Amount Over Time", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# Distribution + category breakdown
row1_col1, row1_col2 = st.columns([1.2, 1])
with row1_col1:
    if amount_col_guess:
        fig = px.histogram(df, x=amount_col_guess, nbins=50, title=f"Distribution of {amount_col_guess}")
        st.plotly_chart(fig, use_container_width=True)
with row1_col2:
    if "category" in df.columns:
        cat_sum = df.groupby("category")[amount_col_guess].sum().reset_index().sort_values(amount_col_guess, ascending=False) if amount_col_guess else df["category"].value_counts().reset_index(names=["category","count"])
        if amount_col_guess:
            fig = px.bar(cat_sum.head(15), x="category", y=amount_col_guess, title="Top Categories by Amount")
        else:
            fig = px.bar(cat_sum.head(15), x="category", y="count", title="Top Categories by Count")
        st.plotly_chart(fig, use_container_width=True)

# Merchant performance
if "merchant" in df.columns and amount_col_guess:
    mcol1, mcol2 = st.columns([1,1])
    with mcol1:
        merch = df.groupby("merchant")[amount_col_guess].agg(["sum","mean","count"]).reset_index().sort_values("sum", ascending=False).head(20)
        fig = px.bar(merch, x="merchant", y="sum", title="Top Merchants by Spend", hover_data=["mean","count"])
        st.plotly_chart(fig, use_container_width=True)
    with mcol2:
        fig = px.scatter(merch, x="count", y="mean", size="sum", hover_name="merchant", title="Merchant Bubble: Count vs Avg Amount")
        st.plotly_chart(fig, use_container_width=True)

# Payment method breakdown
if "payment_method" in df.columns and amount_col_guess:
    pay = df.groupby("payment_method")[amount_col_guess].sum().reset_index().sort_values(amount_col_guess, ascending=False)
    fig = px.pie(pay, names="payment_method", values=amount_col_guess, title="Share by Payment Method")
    st.plotly_chart(fig, use_container_width=True)

# Text insights
if "description" in df.columns:
    st.subheader("📝 Frequent Terms in Descriptions")
    top_words = summarize_text_counts(df, "description", top_n=20)
    fig = px.bar(top_words, x="description", y="count", title="Top Words in Descriptions")
    st.plotly_chart(fig, use_container_width=True)

# Outlier detection
with st.expander("🚨 Outlier Detection (advanced)"):
    if amount_col_guess:
        method = st.radio("Method", ["Isolation Forest", "Z-score (>|3|)"], horizontal=True)
        if method == "Isolation Forest":
            contamination = st.slider("Contamination (expected outlier %)", 0.005, 0.10, 0.03, 0.005)
            mask = detect_outliers(df, amount_col_guess, contamination=contamination)
        else:
            z = st.slider("Z-threshold", 2.0, 5.0, 3.0, 0.1)
            mask = zscore_outliers(df, amount_col_guess, z=z)

        flagged = df[mask]
        st.write(f"Found **{len(flagged)}** potential outliers.")
        if len(flagged) > 0:
            st.dataframe(flagged.head(200), use_container_width=True)
            fig = px.scatter(df, x=amount_col_guess, y=df.index, color=mask.map({True: "Outlier", False: "Normal"}), title="Outlier Scatter (index vs amount)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric amount-like column detected.")

# Mock Insights Chat (simple rules)
with st.expander("🤖 Mock Insights Chat"):
    q = st.text_input("Ask a question about the data (e.g., 'Which category is highest?')")
    if q:
        ql = q.lower()
        answer = None
        if "highest category" in ql or ("highest" in ql and "category" in ql):
            if "category" in df.columns and amount_col_guess:
                topcat = df.groupby("category")[amount_col_guess].sum().idxmax()
                answer = f"Top category by amount is **{topcat}**."
        elif "largest merchant" in ql or ("top merchant" in ql):
            if "merchant" in df.columns and amount_col_guess:
                topm = df.groupby("merchant")[amount_col_guess].sum().idxmax()
                answer = f"Largest merchant by spend is **{topm}**."
        elif "total" in ql:
            if amount_col_guess:
                answer = f"Total amount in the current view: **{df[amount_col_guess].sum():,.2f}**."
        elif "average" in ql or "avg" in ql or "mean" in ql:
            if amount_col_guess:
                answer = f"Average amount in the current view: **{df[amount_col_guess].mean():,.2f}**."
        else:
            answer = "Sorry, I only handle a few patterns in this mock demo (highest category, top merchant, total, average)."
        st.success(answer)

st.caption("Built with Streamlit • Plotly • scikit-learn")
