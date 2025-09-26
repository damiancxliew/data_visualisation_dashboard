import io
import plotly.express as px
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
 
def fig_to_png_bytes(fig, scale=2.0, width=900, height=500):
    """
    Convert a Plotly figure to PNG bytes using Kaleido.
    """
    return fig.to_image(format="png", scale=scale, width=width, height=height, engine="kaleido")

def kpi_table(df, amount_col=None):
    """
    Build a small KPI table (list of lists). Extend as needed.
    """
    rows = [
        ["Metric", "Value"],
        ["Row count", f"{len(df):,}"],
        ["Column count", f"{len(df.columns):,}"],
    ]
    if amount_col and amount_col in df.columns:
        rows += [
            [f"Total {amount_col}", f"{df[amount_col].sum():,.2f}"],
            [f"Average {amount_col}", f"{df[amount_col].mean():,.2f}"],
            [f"P95 {amount_col}", f"{df[amount_col].quantile(0.95):,.2f}"],
        ]
    return rows

def build_standard_figures(df, date_col=None, amount_col=None):
    """
    Create a small set of standard figures for the report.
    """
    figs = []

    # 1) Distribution of amount
    if amount_col and amount_col in df.columns:
        fig_hist = px.histogram(df, x=amount_col, nbins=50, title=f"Distribution of {amount_col}")
        figs.append((f"Distribution of {amount_col}", fig_to_png_bytes(fig_hist)))

    # 2) Top categories by amount or count
    if "category" in df.columns:
        if amount_col and amount_col in df.columns:
            cat_sum = df.groupby("category")[amount_col].sum().reset_index().sort_values(amount_col, ascending=False).head(15)
            fig_cat = px.bar(cat_sum, x="category", y=amount_col, title="Top Categories by Amount")
        else:
            cat_cnt = (
                df["category"]
                .value_counts()
                .reset_index(name="count")  
                .rename(columns={"index": "category"}) 
                .head(10)
            )
            fig_cat = px.bar(cat_cnt, x="category", y="count", title="Top Categories by Count")
        figs.append(("Categories Overview", fig_to_png_bytes(fig_cat)))

    # 3) Pie chart by payment method
    if "payment_method" in df.columns:
        pay_cnt = (
            df["payment_method"]
            .value_counts()
            .reset_index(name="count")       
            .rename(columns={"index": "payment_method"}) 
        )
        fig_pay = px.pie(pay_cnt, names="payment_method", values="count", title="Payment Method Breakdown")
        fig_pay.update_traces(textposition="inside", textinfo="percent+label")
        fig_pay.update_layout(template="plotly_white", showlegend=True)
        figs.append(("Payment Methods", fig_to_png_bytes(fig_pay)))

    return figs

def build_pdf_report(df, *, title="Data Insights Report", date_col=None, amount_col=None, narrative_text=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceAfter=6))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=4))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, textColor=colors.grey))

    elements = []

    # Header
    elements.append(Paragraph(title, styles["H1"]))
    elements.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Small"]))
    elements.append(Spacer(1, 10))

    # KPI table
    elements.append(Paragraph("Summary", styles["H2"]))
    table_data = kpi_table(df, amount_col=amount_col)
    tbl = Table(table_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f2f6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafafa")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 12))

    if narrative_text:
        elements.append(Paragraph("Executive Summary", styles["H2"]))
        for para in narrative_text.split("\n")[1:]:
            if para.strip():
                elements.append(Paragraph(para.strip().replace("**", ""), styles["Normal"]))
        elements.append(Spacer(1, 8))

    # Standard charts
    figs = build_standard_figures(df, date_col=date_col, amount_col=amount_col)
    for caption, png_bytes in figs:
        elements.append(Paragraph(caption, styles["H2"]))
        img = RLImage(io.BytesIO(png_bytes), width=500, height=280)  # auto-scale as you like
        elements.append(img)
        elements.append(Spacer(1, 10))

    doc.build(elements)
    return buf.getvalue()