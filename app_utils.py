
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.ensemble import IsolationForest

DATE_COL_CANDIDATES = ["date", "timestamp", "datetime"]

def load_dataset(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    # best-effort parse date-like columns
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in DATE_COL_CANDIDATES):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    # normalize string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def basic_stats(df: pd.DataFrame) -> Dict:
    stats = {}
    stats["rows"], stats["cols"] = df.shape
    stats["numeric_cols"] = df.select_dtypes(include=np.number).columns.tolist()
    stats["categorical_cols"] = df.select_dtypes(exclude=np.number).columns.tolist()
    return stats

def detect_outliers(df: pd.DataFrame, column: str, contamination: float = 0.03, random_state: int = 42):
    """Return a boolean mask of outliers using IsolationForest on a single numeric column."""
    x = df[[column]].dropna()
    if x.empty:
        return pd.Series([False] * len(df), index=df.index)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(x.values)
    preds = model.predict(x.values)  # -1 outlier, 1 inlier
    outlier_idx = x.index[preds == -1]
    mask = pd.Series([False] * len(df), index=df.index)
    mask.loc[outlier_idx] = True
    return mask

def zscore_outliers(df: pd.DataFrame, column: str, z: float = 3.0):
    x = df[column]
    if x.isna().all():
        return pd.Series([False] * len(df), index=df.index)
    mu = x.mean()
    sd = x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0
    mask = (np.abs((x - mu) / sd) > z)
    return mask.fillna(False)

def summarize_text_counts(df: pd.DataFrame, text_col: str, top_n: int = 20):
    tokens = (
        df[text_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.split()
        .explode()
        .dropna()
    )
    vc = tokens.value_counts().head(top_n)
    # Give the index a proper name, and the counts a simple string column name
    return vc.rename_axis(text_col).reset_index(name="count")
