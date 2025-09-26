# 💳 Financial Transactions – Interactive Dashboard

An **interactive Streamlit dashboard** to explore and analyze a financial transactions dataset with **dynamic charting, advanced analytics, and AI-powered insights**.

🌐 **Live Demo:** [data-visualisation-dashboard.streamlit.app](https://data-visualisation-dashboard.streamlit.app/)

---

## 🚀 Features

-   **Data Loading**

    -   Upload any CSV file or use the bundled `financial_transactions.csv`
    -   Auto-detects **date** and **amount** columns by name

-   **Filters**

    -   Global sidebar filters for **date range**, **numeric ranges**, and **categories**
    -   Apply filters dynamically across all charts

-   **KPIs / Quick Insights**

    -   Total rows, total columns, #numeric vs. #categorical
    -   Sum, mean, median displayed as metrics

-   **Custom Chart Builder**

    -   **Single Chart Builder**: create one visualization at a time
    -   **Multi-Chart Dashboard**: build and save multiple charts, rendered side by side
    -   Supported chart types:
        `Bar`, `Line`, `Scatter`, `Pie`, `Box`, `Histogram`, `Heatmap`

-   **Quick Explore**

    -   One-click exploration of **numeric distributions** and **categorical breakdowns**

-   **Advanced Analytics**

    -   **Outlier detection** (Z-Score, IQR, Isolation Forest)
    -   **Statistical summary** of numeric columns

-   **Export**

    -   **Download Filtered Data** as CSV
    -   **Generate PDF Report** with:

        -   Summary statistics
        -   Category/merchant breakdowns
        -   AI-generated narrative (optional if API key provided)

-   **AI Insights**

    -   **Chatbot** powered by OpenAI (`gpt-4o-mini` by default)
    -   Context-aware: analyzes the **currently filtered dataset**
    -   Provides actionable insights, anomaly detection, and recommendations
    -   Pre-loaded with suggested starter questions

-   **Styling**

    -   External CSS (`styles/styles.css`) for consistent theme

---

## 📂 Dataset

Place your dataset as `financial_transactions.csv` alongside `app.py`, or upload via the UI.

**Expected columns (auto-detected by heuristics, flexible names allowed):**

-   `transaction_id` (string)
-   `date` (datetime-like)
-   `amount` (numeric)
-   `category`, `merchant`, `payment_method`, `account_type`, `transaction_type` (categorical)
-   `description` (text for NLP insights)

---

## ▶️ Quickstart

Try it online without installing anything:  
👉 [Live Demo on Streamlit Cloud](https://data-visualisation-dashboard.streamlit.app/)

Or run locally:

1. **Clone & Setup Environment**

    ```bash
    git clone https://github.com/damiancxliew/data_visualisation_dashboard.git
    cd data_visualisation_dashboard
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run Streamlit**

    ```bash
    streamlit run app.py
    ```

---

## 🔑 AI Features

To enable AI insights (chatbot + PDF narrative):

1. Get an **OpenAI API key** from [platform.openai.com](https://platform.openai.com/)
2. Add it in the Streamlit sidebar under **AI Configuration**

> If no key is provided, the dashboard still runs — you’ll just miss the AI features.

---

## 📘 Tech Stack

-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **Visualization**: Plotly, Seaborn, Matplotlib
-   **ML/Analytics**: Scikit-learn (Isolation Forest), NumPy, Pandas
-   **AI**: OpenAI GPT models (via `openai` package)
-   **PDF Export**: ReportLab

---
