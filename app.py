import pandas as pd
import plotly.express as px
import streamlit as st

from src.categorizer import categorize_transactions
from src.data_loader import load_transactions_csv
from src.preprocessing import preprocess_transactions


st.set_page_config(
    page_title="Personal Finance Insight",
    page_icon="💳",
    layout="wide",
)


def process_uploaded_file(uploaded_file):
    """Clean and categorize one uploaded Chase statement."""
    raw_df = load_transactions_csv(uploaded_file)
    clean_df = preprocess_transactions(raw_df)

    # Synthetic data contains ground-truth labels. Drop them so both synthetic
    # and real statements go through the same production model.
    clean_df = clean_df.drop(columns=["Category"], errors="ignore")

    categorized_df = categorize_transactions(clean_df)
    categorized_df["Source File"] = uploaded_file.name
    return categorized_df


def combine_uploaded_files(uploaded_files):
    """Process valid files and collect user-facing errors separately."""
    dataframes = []
    errors = []

    for uploaded_file in uploaded_files:
        try:
            dataframes.append(process_uploaded_file(uploaded_file))
        except Exception as error:
            errors.append(f"Could not process {uploaded_file.name}: {error}")

    if not dataframes:
        return pd.DataFrame(), errors

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df["Month"] = combined_df["Date"].dt.to_period("M").astype(str)
    return combined_df, errors


def prepare_month_data(transactions, selected_month):
    """Return all monthly transactions and spending-only transactions."""
    monthly_df = transactions[
        transactions["Month"] == selected_month
    ].copy()

    expenses_df = monthly_df[
        (monthly_df["Amount"] < 0)
        & (monthly_df["Category"] != "Transfer")
    ].copy()
    expenses_df["Spend"] = -expenses_df["Amount"]

    return monthly_df, expenses_df


def summarize_spending(expenses):
    """Calculate category totals used by metrics and future charts."""
    category_spending = (
        expenses
        .groupby("Category", as_index=False)["Spend"]
        .sum()
        .sort_values("Spend", ascending=False)
        .reset_index(drop=True)
    )

    total_spending = expenses["Spend"].sum()
    transaction_count = len(expenses)
    top_category = (
        category_spending.iloc[0]["Category"]
        if not category_spending.empty
        else "N/A"
    )

    return {
        "category_spending": category_spending,
        "total_spending": total_spending,
        "transaction_count": transaction_count,
        "top_category": top_category,
    }


def render_spending_chart(category_spending, selected_month):
    """Render the category spending donut chart and source data."""
    st.subheader(f"Spending by category — {selected_month}")

    if category_spending.empty:
        st.info("No expense transactions found for this month.")
        return

    figure = px.pie(
        category_spending,
        names="Category",
        values="Spend",
        hole=0.4,
    )
    figure.update_traces(
        textinfo="none",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Spending: $%{value:,.2f}<br>"
            "Share: %{percent:.1%}"
            "<extra></extra>"
        ),
    )
    figure.update_layout(
        margin={"t": 10, "r": 10, "b": 10, "l": 10},
        legend_title_text="Category",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key=f"spending-chart-{selected_month}",
    )

    with st.expander("View category totals"):
        st.dataframe(
            category_spending,
            use_container_width=True,
            hide_index=True,
        )


def main():
    st.title("Personal Finance Insight")
    st.write(
        "Upload one or more Chase CSV statements to analyze your spending."
    )

    uploaded_files = st.file_uploader(
        "Upload Chase statements",
        type=["csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one CSV statement to begin.")
        st.stop()

    combined_df, errors = combine_uploaded_files(uploaded_files)

    for error in errors:
        st.error(error)

    if combined_df.empty:
        st.error("None of the uploaded files could be processed.")
        st.stop()

    st.success(
        f"Loaded {len(combined_df):,} transactions "
        f"from {len(uploaded_files) - len(errors)} file(s)."
    )

    available_months = sorted(
        combined_df["Month"].unique(),
        reverse=True,
    )

    st.subheader("Dashboard")
    selected_month = st.selectbox(
        "Select a month",
        options=available_months,
    )

    monthly_df, expenses_df = prepare_month_data(
        combined_df,
        selected_month,
    )
    summary = summarize_spending(expenses_df)

    st.caption(
        f"Showing {len(monthly_df):,} transactions for {selected_month}."
    )

    render_spending_chart(
        summary["category_spending"],
        selected_month,
    )

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric(
        "Total spending",
        f"${summary['total_spending']:,.2f}",
    )
    metric_2.metric(
        "Expense transactions",
        f"{summary['transaction_count']:,}",
    )
    metric_3.metric("Top category", summary["top_category"])

    st.subheader("Uploaded data preview")
    st.dataframe(
        monthly_df.head(50),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
