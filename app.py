import pandas as pd
import streamlit as st

from src.dashboard_components import (
    render_month_comparison,
    render_spending_chart,
)
from src.spending_analysis import (
    analyze_month,
    compare_spending_summaries,
)
from src.transaction_pipeline import combine_uploaded_files


st.set_page_config(
    page_title="Personal Finance Insight",
    page_icon="💳",
    layout="wide",
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

    comparison_month = None

    if len(available_months) > 1:
        enable_comparison = st.toggle(
            "Compare with another month"
        )

        if enable_comparison:
            comparison_options = [
                month for month in available_months if month != selected_month
            ]
            previous_month = str(
                pd.Period(selected_month, freq="M") - 1
            )

            default_comparison = (
                previous_month
                if previous_month in comparison_options
                else comparison_options[0]
            )

            comparison_month = st.selectbox(
                "Compare with",
                options=comparison_options,
                index=comparison_options.index(default_comparison),
            )
    else:
        st.caption(
            "Upload transactions from another month "
            "to enable comparison."
        )

    current_analysis = analyze_month(
        combined_df,
        selected_month,
    )
    monthly_df = current_analysis["transactions"]
    summary = current_analysis["summary"]

    comparison = None

    if comparison_month is not None:
        comparison_analysis = analyze_month(
            combined_df,
            comparison_month,
        )

        comparison = compare_spending_summaries(
            summary,
            comparison_analysis["summary"],
        )

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

    render_month_comparison(
        comparison,
        selected_month,
        comparison_month,
    )

    st.subheader("Uploaded data preview")
    st.dataframe(
        monthly_df.head(50),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
