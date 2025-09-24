# Financial Transactions – Interactive Dashboard

An interactive Streamlit dashboard to explore the provided financial transactions dataset.

## 📦 Features

-   Sidebar filters for date, amount range, category, merchant, payment method, account/account type, transaction type.
-   KPI tiles (count, total, average, categories).
-   Time series (daily sums), histograms, category & merchant breakdowns, payment-method pie.
-   Text analytics: top words in `description`.
-   **Advanced:** Outlier detection (Isolation Forest or Z-score) and a simple mock insights chat.

## 🗂 Dataset

Place your CSV as `financial_transactions.csv` in the project root, or upload via the UI. The app also auto-detects the mounted path `/mnt/data/financial_transactions.csv` if present.

### Expected columns

-   `transaction_id` (string)
-   `date` (datetime-like)
-   `amount` (numeric)
-   `category`, `merchant`, `payment_method`, `account_type`, `transaction_type` (categorical)
-   `description` (text)

> Column names are flexible; the app tries to infer date and amount columns.

## ▶️ How to run

1. **Create & activate a virtual environment (recommended)**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Start the app**

    ```bash
    streamlit run app.py
    ```

4. **Open your browser**
   Streamlit will print a local URL (e.g., http://localhost:8501).

## 🧱 Project Structure

```
finviz_dashboard/
├── app.py
├── app_utils.py
├── requirements.txt
└── README.md
```

## ⚙️ Notes

-   The outlier detector uses `IsolationForest` on the chosen numeric column (by default `amount`). Adjust the contamination slider to tune sensitivity.
-   The mock insights chat is a simple rule-based demo; consider replacing with a retrieval or LLM-backed agent in production.
-   Keep code modular in `app_utils.py` for extendability (e.g., add forecasting, segment analysis, seasonality decomposition).

## 🔐 Troubleshooting

-   **No date filter**: Ensure your `date` column parses as datetime. If not, pre-parse or rename to include `date`/`time`.
-   **No charts showing**: Verify `amount` is numeric; remove currency symbols before loading.
-   **Large files**: Consider enabling `@st.cache_data` and sampling for heavy charts.

---

Built for quick insight discovery and extensibility.
