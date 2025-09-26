import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from utils.app_utils import load_dataset, basic_stats, detect_outliers, zscore_outliers, summarize_text_counts
from utils.report_utils import build_pdf_report
from utils.ai_insights_utils import gen_narrative_with_openai, _profile_dataframe_for_llm, _build_messages, call_llm_openai
from utils.chart_utils import create_dynamic_chart, chart_builder_ui
from pathlib import Path
from datetime import datetime
from openai import OpenAI

def load_css(path):
    css_path = Path(path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {css_path}")

st.set_page_config(page_title="Advanced Data Explorer", layout="wide", initial_sidebar_state="expanded")

load_css('styles/styles.css')


@st.cache_data(show_spinner=False)
def _load(path, mtime):
    return load_dataset(path)

def pick_default_path():
    candidates = [
        "financial_transactions.csv",
        "/mnt/data/financial_transactions.csv"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


# Initialise session state for dynamic charts
if 'chart_configs' not in st.session_state:
    st.session_state.chart_configs = []

if "multicharts" not in st.session_state:
    st.session_state.multicharts = {}


# Main App Header
st.title("Advanced Data Explorer Dashboard")
# st.markdown("*Build interactive visualizations like Tableau with drag-and-drop simplicity*")

# Data loading section
with st.sidebar:
    st.header("📁 Data Source")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    default_path = pick_default_path()
    
    if default_path:
        st.caption(f"Default: {default_path}")
        use_default = st.checkbox("Use default dataset", value=True)
    else:
        use_default = False

# Load data
if uploaded is not None:
    df = load_dataset(uploaded)
    st.success(f"✅ Loaded {len(df)} rows from uploaded file")
elif use_default and default_path:
    mtime = os.path.getmtime(default_path)
    df = _load(default_path, mtime)
    st.success(f"✅ Loaded {len(df)} rows from default dataset")
else:
    st.info("🔺 Please upload a CSV file to get started")
    st.stop()

# Data Overview
with st.expander("📋 Dataset Overview", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns):,}")
    with col3:
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", f"{numeric_count}")
    with col4:
        categorical_count = len(df.select_dtypes(include=['object']).columns)
        st.metric("Text Columns", f"{categorical_count}")
    
    # Show first few rows
    st.subheader("Sample Data")
    st.dataframe(df.head(), use_container_width=True)

# Enhanced Filters in Sidebar
with st.sidebar:
    st.header("🔍 Global Filters")
    
    # Get column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Auto-detect common column patterns
    date_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['date', 'time', 'created', 'updated']):
            try:
                df[col] = pd.to_datetime(df[col])
                date_col = col
                break
            except:
                continue
    
    amount_col = None
    for col in numeric_cols:
        if any(term in col.lower() for term in ['amount', 'value', 'price', 'cost', 'total']):
            amount_col = col
            break
    
    # Date filter
    if date_col:
        st.subheader("📅 Date Range")
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        date_range = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)
            df = df[mask]
    
    # Numeric filters
    if numeric_cols:
        st.subheader("📊 Numeric Filters")
        for col in numeric_cols[:3]:  # Limit to avoid clutter
            if df[col].nunique() > 10:  # Only show slider for columns with many unique values
                min_val, max_val = float(df[col].min()), float(df[col].max())
                if min_val != max_val:
                    range_vals = st.slider(
                        f"{col}",
                        min_val, max_val, (min_val, max_val),
                        key=f"slider_{col}"
                    )
                    df = df[(df[col] >= range_vals[0]) & (df[col] <= range_vals[1])]
    
    # Categorical filters
    if categorical_cols:
        st.subheader("🏷️ Category Filters")
        for col in categorical_cols[:5]:  # Limit to avoid clutter
            unique_vals = sorted(df[col].dropna().astype(str).unique())
            if len(unique_vals) <= 50:  # Only show multiselect for manageable number of options
                selected = st.multiselect(
                    f"{col}",
                    unique_vals,
                    key=f"filter_{col}"
                )
                if selected:
                    df = df[df[col].astype(str).isin(selected)]

# Main Dashboard Area
st.markdown("---")

# Quick Insights Section
st.header("⚡ Quick Insights")

if numeric_cols:
    insight_cols = st.columns(len(numeric_cols) if len(numeric_cols) <= 4 else 4)
    for i, col in enumerate(numeric_cols[:4]):
        with insight_cols[i]:
            col_sum = df[col].sum()
            col_mean = df[col].mean()
            col_median = df[col].median()
            
            st.metric(
                label=f"Total {col}",
                value=f"{col_sum:,.2f}",
                delta=f"Avg: {col_mean:.2f}"
            )

# Chart Builder Section
st.markdown("---")
st.header("🎨 Custom Chart Builder")

# Tabs for different chart builders
tab1, tab2, tab3 = st.tabs(["Single Chart", "Multi-Chart Dashboard", "Quick Explore"])

with tab1:
    st.markdown("### Build Your Custom Visualization")
    cfg, submitted = chart_builder_ui(df, 0)
    if submitted:
        # Validate columns exist before trying to render
        chart_type = cfg.get("type")
        if chart_type in ["Bar Chart", "Line Chart"]:
            if not (cfg.get("x_axis") and cfg.get("y_axis")):
                st.warning("Please select both X and Y axes.")
            else:
                fig = create_dynamic_chart(df, cfg)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        elif chart_type in ["Scatter Plot"]:
            if not (cfg.get("x_axis") and cfg.get("y_axis")):
                st.warning("Please select both X and Y axes.")
            else:
                fig = create_dynamic_chart(df, cfg)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        elif chart_type in ["Pie Chart"]:
            if not cfg.get("category_column"):
                st.warning("Please select a category column.")
            else:
                fig = create_dynamic_chart(df, cfg)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        elif chart_type in ["Box Plot"]:
            if not cfg.get("y_axis"):
                st.warning("Please select a numeric Y-axis.")
            else:
                fig = create_dynamic_chart(df, cfg)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        elif chart_type in ["Histogram"]:
            if not cfg.get("column"):
                st.warning("Please select a numeric column.")
            else:
                fig = create_dynamic_chart(df, cfg)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        elif chart_type in ["Heatmap"]:
            if not (cfg.get("x_axis") and cfg.get("y_axis") and cfg.get("value_column")):
                st.warning("Please select X, Y, and Values columns.")
            else:
                fig = create_dynamic_chart(df, cfg)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Create Multiple Charts")

    left, right = st.columns([1, 1])
    with left:
        num_charts = st.slider("Number of builders", 1, 4, 2, key="num_builders")
    with right:
        if st.button("🧹 Clear all charts", use_container_width=True):
            st.session_state.multicharts = {}
            st.success("Cleared all saved charts.")

    # Show builders
    builder_cols = st.columns(2)
    for i in range(num_charts):
        with builder_cols[i % 2]:
            cfg, submitted = chart_builder_ui(df, i + 10)  # offset IDs to avoid key clashes
            if submitted:
                # Save/overwrite this chart's config
                st.session_state.multicharts[i] = cfg
                st.toast(f"Chart #{i+1} saved", icon="✅")

    st.markdown("---")
    st.markdown("### Rendered Charts")

    if not st.session_state.multicharts:
        st.info("No charts saved yet. Use the builders above to create charts.")
    else:
        render_cols = st.columns(2)
        # Sort by chart_id to keep stable order
        for idx, (cid, cfg) in enumerate(sorted(st.session_state.multicharts.items())):
            fig = create_dynamic_chart(df, cfg)
            if fig:
                with render_cols[idx % 2]:
                    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Quick Data Exploration")
    
    explore_col1, explore_col2, explore_col3 = st.columns(3)
    
    with explore_col1:
        st.subheader("🔢 Numeric Distributions")
        if numeric_cols:
            selected_numeric = st.selectbox("Select numeric column", numeric_cols)
            if selected_numeric:
                fig = px.histogram(df, x=selected_numeric, marginal="box", 
                                 title=f"Distribution of {selected_numeric}")
                st.plotly_chart(fig, use_container_width=True)
    
    with explore_col2:
        st.subheader("🏷️ Category Breakdowns")
        if categorical_cols:
            selected_categorical = st.selectbox("Select categorical column", categorical_cols)
            if selected_categorical:
                value_counts = df[selected_categorical].value_counts().head(10)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Top 10 {selected_categorical}")
                st.plotly_chart(fig, use_container_width=True)

    with explore_col3:
        # NLP/Text Insights Section
        st.subheader("📝 Text Insights\n")
        try:
            word_counts = summarize_text_counts(df, 'description', top_n=20)
            # st.caption(f"Top Words in '{selected_text_col}'")
            fig = px.bar(
                word_counts,
                x='description',
                y="count",
                title=f"Most Frequent Words in Description",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(word_counts, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing text column: {e}")
 


# Advanced Analytics Section
st.markdown("---")
st.header("Advanced Analytics")

advanced_tabs = st.tabs(["Outliers", "Statistical Summary"])

with advanced_tabs[0]:
    if numeric_cols:
        st.subheader("Outlier Detection")
        outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)
        method = st.radio("Detection method", ["Z-Score", "IQR", "Isolation Forest"], horizontal=True)
        
        # Create outlier mask
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        if method == "Z-Score":
            threshold = st.slider("Z-Score threshold", 2.0, 4.0, 3.0, 0.1)
            z_scores = np.abs((df[outlier_col] - df[outlier_col].mean()) / df[outlier_col].std())
            outlier_mask = z_scores > threshold
            outliers = df[outlier_mask]
        elif method == "IQR":
            Q1 = df[outlier_col].quantile(0.25)
            Q3 = df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)
            outliers = df[outlier_mask]
        elif method == "Isolation Forest":
            contamination = st.slider("Contamination (expected outlier %)", 0.005, 0.20, 0.05, 0.005)
            outlier_mask = detect_outliers(df, outlier_col, contamination=contamination)
            outliers = df[outlier_mask]
        
        st.write(f"Found **{len(outliers)}** outliers ({len(outliers)/len(df)*100:.2f}%)")
        
        # Visualization section
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Box plot showing outliers
            fig_box = px.box(df, y=outlier_col, title=f"Box Plot - {outlier_col}")
            fig_box.update_traces(
                marker_color='lightblue',
                marker_outliercolor='red',
                marker_size=8
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with viz_col2:
            # Scatter plot with outliers highlighted
            df_viz = df.copy()
            df_viz['outlier_status'] = outlier_mask.map({True: 'Outlier', False: 'Normal'})
            
            fig_scatter = px.scatter(
                df_viz,
                x=df_viz.index,
                y=outlier_col,
                color='outlier_status',
                color_discrete_map={'Normal': 'lightblue', 'Outlier': 'red'},
                title=f"Outlier Detection - {outlier_col}",
                labels={'x': 'Index', 'y': outlier_col}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Distribution with outliers highlighted
        fig_hist = px.histogram(
            df_viz, 
            x=outlier_col, 
            color='outlier_status',
            color_discrete_map={'Normal': 'lightblue', 'Outlier': 'red'},
            title=f"Distribution with Outliers - {outlier_col}",
            marginal="rug"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Show outlier data
        if len(outliers) > 0:
            st.subheader("Outlier Data")
            st.dataframe(outliers.head(50), use_container_width=True)

with advanced_tabs[1]:
    st.subheader("Statistical Summary")
    if numeric_cols:
        summary_stats = df[numeric_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)

# Data Export Section
st.markdown("---")
st.header("💾 Export Options")

with st.sidebar:
    st.subheader("🤖 AI Configuration")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Paste a key with access to GPT-4o/GPT-4o-mini or compatible.",
        key="openai_key"
    )
    
    if openai_api_key:
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ Add API key to enable AI insights")


export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    if st.button("1) Download Filtered Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,       
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with export_col2:
    if st.button("2) Generate Report"):
        try:
            # Detect commonly used columns
            date_col_detected = None
            for c in df.columns:
                if any(k in c.lower() for k in ["date", "time", "created", "updated"]):
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col_detected = c
                        break

            amount_col_detected = None
            for c in df.select_dtypes(include=[np.number]).columns:
                if any(k in c.lower() for k in ["amount", "value", "price", "cost", "total"]):
                    amount_col_detected = c
                    break

            narrative_text = gen_narrative_with_openai(df, date_col_detected, amount_col_detected, api_key=openai_api_key) 

            pdf_bytes = build_pdf_report(
                df,
                title="Advanced Data Explorer – Insights Report",
                date_col=date_col_detected,
                amount_col=amount_col_detected,
                narrative_text=narrative_text
            )

            st.success("✅ Report generated")
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"insights_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")


# =========================
# AI Insights Chatbot Integration
# =========================

# Add the chatbot section
st.markdown("---")
st.header("AI-Powered Insights")

st.caption("Ask questions about your filtered data. The AI will analyse the current view and provide actionable insights.")

# Initialize chat history
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

# Chat interface
chat_container = st.container()

# Suggested questions
if not st.session_state.ai_chat_history:
    st.subheader("💡 Suggested Questions")
    suggestion_cols = st.columns(2)
    
    suggestions = [
        "What are the key trends in this data?",
        "Identify the top 3 insights from this dataset",
        "What anomalies or outliers should I investigate?",
        "How can I improve performance based on this data?",
        "What additional analysis would you recommend?",
        "Summarize the main patterns you see"
    ]
    
    for i, suggestion in enumerate(suggestions):
        col = suggestion_cols[i % 2]
        with col:
            if st.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.ai_chat_history.append({"role": "user", "content": suggestion})

# Chat input
with st.form("ai_chat_form", clear_on_submit=True):
    user_question = st.text_area(
        "Ask a question about your data:",
        placeholder="e.g., What patterns do you see in the spending data?",
        height=100
    )
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        send_button = st.form_submit_button("Send 🚀", use_container_width=True)
    with col2:
        clear_button = st.form_submit_button("Clear Chat 🗑️", use_container_width=True)

if clear_button:
    st.session_state.ai_chat_history = []
    st.rerun()

if send_button and user_question.strip():
    st.session_state.ai_chat_history.append({"role": "user", "content": user_question.strip()})

# Display chat history
with chat_container:
    for i, message in enumerate(st.session_state.ai_chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Generate AI response
if st.session_state.ai_chat_history and st.session_state.ai_chat_history[-1]["role"] == "user":
    if not openai_api_key:
        with st.chat_message("assistant"):
            st.warning("🔑 Please add your OpenAI API key in the sidebar to get AI insights.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analysing your data..."):
                try:
                    # Get the amount column for context
                    amount_col = None
                    numeric_cols_current = df.select_dtypes(include=[np.number]).columns.tolist()
                    for col in numeric_cols_current:
                        if any(term in col.lower() for term in ['amount', 'value', 'price', 'cost', 'total']):
                            amount_col = col
                            break
                    
                    context = _profile_dataframe_for_llm(df, amount_col, date_col)
                    messages = _build_messages(context, st.session_state.ai_chat_history)
                    
                    ai_response = call_llm_openai(messages, openai_api_key, model="gpt-4o-mini", temperature=0.3)
                    
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": ai_response})
                    st.markdown(ai_response)
                    
                except Exception as e:
                    error_msg = f"❌ Error generating AI insights: {str(e)}"
                    st.error(error_msg)
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("*Data Explorer Dashboard • Damian Liew Cho Xiang*")