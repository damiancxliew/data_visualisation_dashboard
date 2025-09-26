import pandas as pd
import numpy as np
from openai import OpenAI

def gen_narrative_with_openai(df, date_col=None, amount_col=None, api_key=None):
    if not api_key:
        return None
    # Build a tiny context
    parts = [f"Rows: {len(df)}", f"Columns: {', '.join(map(str, df.columns[:15]))}"]
    if amount_col and amount_col in df.columns:
        parts.append(f"Total {amount_col}: {df[amount_col].sum():,.2f}")
        parts.append(f"Mean {amount_col}: {df[amount_col].mean():,.2f}")
    if date_col and date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        parts.append(f"Date range: {df[date_col].min().date()} to {df[date_col].max().date()}")
    context = "\n".join(parts)

    client = OpenAI(api_key=api_key)
    prompt = f"""You are a data analyst. Based on the context below, draft a concise executive summary with 3–5 bullet points and a one-line recommendation.
    Context:
    {context}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=250,
    )
    return resp.choices[0].message.content


def _profile_dataframe_for_llm(df_view, amount_col_guess = None, date_col = None) -> str:
    """Profile the current dataframe for the LLM"""
    parts = []
    parts.append(f"Dataset Overview: {len(df_view)} rows, {len(df_view.columns)} columns")
    parts.append(f"Columns: {', '.join(map(str, df_view.columns))}")
    
    # Numeric column insights
    numeric_cols = df_view.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            if len(df_view) > 0:
                total = float(df_view[col].sum())
                mean = float(df_view[col].mean())
                std = float(df_view[col].std()) if len(df_view) > 1 else 0
                parts.append(f"{col} stats: total={total:,.2f}, mean={mean:,.2f}, std={std:,.2f}")
    
    # Date range info
    if date_col and np.issubdtype(df_view[date_col].dtype, np.datetime64):
        if len(df_view) > 0:
            dmin = pd.to_datetime(df_view[date_col]).min()
            dmax = pd.to_datetime(df_view[date_col]).max()
            parts.append(f"Date range: {dmin.date()} to {dmax.date()}")
    
    # Categorical insights
    categorical_cols = df_view.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols[:5]:  # Top 5 categorical columns
        if len(df_view) > 0:
            top_vals = df_view[col].value_counts().head(3).to_dict()
            parts.append(f"Top {col} values: {top_vals}")
    
    # Cross-tabulations for insights
    if amount_col_guess and len(categorical_cols) > 0:
        for col in categorical_cols[:2]:  # Top 2 categories
            if len(df_view) > 0:
                top_amt = (
                    df_view.groupby(col)[amount_col_guess].sum()
                    .sort_values(ascending=False).head(3).to_dict()
                )
                parts.append(f"Top {col} by {amount_col_guess}: {top_amt}")
    
    return "\n".join(parts)

def _build_messages(context: str, history: list[dict]):
    """Build message history for OpenAI API"""
    system_prompt = f"""
You are an expert data analyst embedded in an interactive dashboard. The user is exploring a filtered dataset.

Your role:
1. Analyze the provided dataset context
2. Answer questions with specific, actionable insights
3. Suggest data-driven recommendations
4. Identify patterns, trends, and anomalies
5. Propose next steps for analysis

Dataset Context:
{context}

Guidelines:
- Be specific and quantitative when possible
- Use bullet points for clarity
- Suggest actionable next steps (max 3)
- If asked about calculations, show your reasoning
- Focus on business/practical implications
- Keep responses concise but insightful
- If data seems insufficient, explain what additional data would help
"""
    
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(history[-10:])  # Keep last 10 messages for context
    return msgs

def call_llm_openai(messages: list[dict], api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3) -> str:
    """Call OpenAI API"""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1000
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"