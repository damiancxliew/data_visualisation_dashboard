# Financial Transactions – Interactive Dashboard

An interactive Streamlit dashboard to explore a financial transactions dataset.

## 📦 Features

-   **Filters**: date range, amount range, category, merchant, payment method, account type, transaction type
-   **KPIs**: #transactions, total amount, average amount, #categories
-   **Charts**: daily time series, amount histogram, top categories/merchants, payment-method pie
-   **Text Insights**: top words from `description`
-   **Outlier Detection**: outlier detection (Isolation Forest / Z-score)
-   **AI Insights**: OpenAI-powered **Insights Chatbot** that uses the **current filtered view** to suggest actionable insights

## 🗂 Dataset

Place your CSV as `financial_transactions.csv` alongside `app.py`, or upload via the UI.

**Expected columns (flexible):**

-   `transaction_id` (string)
-   `date` (datetime-like)
-   `amount` (numeric)
-   `category`, `merchant`, `payment_method`, `account_type`, `transaction_type` (categorical)
-   `description` (text)

> The app auto-detects date/amount columns by name heuristics.

## ▶️ Quickstart

1. **(Recommended) Create a virtual environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    ```
