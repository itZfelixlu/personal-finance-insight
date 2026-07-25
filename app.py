import pandas as pd
import streamlit as st

from src.ui.dashboard_components import (
    render_month_comparison,
    render_spending_chart,
    render_weekly_patterns,
)
from llm.insight_generator import (
    generate_insight,
    prepare_insight_for_markdown,
)
from src.ml.pattern_predictor import predict_patterns
from src.ui.profile_components import (
    profile_context_key,
    render_user_profile,
)
from src.pipeline.spending_analysis import (
    analyze_month,
    compare_spending_summaries,
)
from src.pipeline.transaction_pipeline import combine_uploaded_files
from src.ml.weekly_features import build_weekly_features


st.set_page_config(
    page_title="Personal Finance Insight",
    page_icon="💳",
    layout="wide",
)


def main():
    # Page introduction and Chase statement upload
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

    # Uploaded-file processing, validation, and duplicate-file feedback
    combined_df, errors, warnings = combine_uploaded_files(uploaded_files)

    for error in errors:
        st.error(error)

    for warning in warnings:
        st.warning(warning)

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

    # Dashboard month selection and optional month-to-month comparison
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

    # Optional user profile and monthly context for personalized AI insights
    user_profile = render_user_profile(selected_month)

    # Current-month analysis and optional comparison calculations
    current_analysis = analyze_month(
        combined_df,
        selected_month,
    )
    monthly_df = current_analysis["transactions"]
    summary = current_analysis["summary"]

    comparison = None
    comparison_analysis = None

    if comparison_month is not None:
        comparison_analysis = analyze_month(
            combined_df,
            comparison_month,
        )

        comparison = compare_spending_summaries(
            summary,
            comparison_analysis["summary"],
        )

    # Spending chart, summary metrics, and comparison dashboard
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

    # AI insight controls and cache context for the current selections
    st.subheader("AI Financial Insight")
    st.caption(
        "Generate a personalized explanation of the current dashboard. "
        "The AI receives summarized analysis rather than raw transactions."
    )

    insight_context = (
        selected_month,
        comparison_month,
        profile_context_key(user_profile),
    )

    if st.button("Generate AI Insight", type="primary"):
        try:
            # Weekly pattern prediction and personalized LLM insight generation
            weekly_patterns = predict_patterns(
                build_weekly_features(monthly_df)
            )
            comparison_weekly_patterns = None

            if comparison_analysis is not None:
                comparison_weekly_patterns = predict_patterns(
                    build_weekly_features(
                        comparison_analysis["transactions"]
                    )
                )

            st.session_state["weekly_patterns"] = weekly_patterns
            st.session_state[
                "comparison_weekly_patterns"
            ] = comparison_weekly_patterns
            st.session_state[
                "weekly_patterns_context"
            ] = insight_context

            with st.spinner("Generating AI insight..."):
                insight = generate_insight(
                    api_key=st.secrets["OPENAI_API_KEY"],
                    selected_month=selected_month,
                    summary=summary,
                    weekly_patterns=weekly_patterns,
                    comparison=comparison,
                    comparison_month=comparison_month,
                    comparison_weekly_patterns=(
                        comparison_weekly_patterns
                    ),
                    user_profile=user_profile,
                    model=st.secrets.get(
                        "OPENAI_MODEL",
                        "gpt-5.6-luna",
                    ),
                )

            st.session_state["ai_insight"] = insight
            st.session_state["ai_insight_context"] = insight_context
        except KeyError:
            st.error(
                "OPENAI_API_KEY is missing from "
                ".streamlit/secrets.toml."
            )
        except Exception as error:
            st.error(f"Could not generate AI insight: {error}")

    # Display cached pattern predictions and AI insight without another API call
    saved_insight = st.session_state.get("ai_insight")
    saved_context = st.session_state.get("ai_insight_context")
    saved_patterns = st.session_state.get("weekly_patterns")
    saved_comparison_patterns = st.session_state.get(
        "comparison_weekly_patterns"
    )
    saved_patterns_context = st.session_state.get(
        "weekly_patterns_context"
    )

    if (
        saved_patterns is not None
        and saved_patterns_context == insight_context
    ):
        render_weekly_patterns(
            saved_patterns,
            selected_month,
            saved_comparison_patterns,
            comparison_month,
        )

    if saved_insight and saved_context == insight_context:
        st.markdown(prepare_insight_for_markdown(saved_insight))
    elif saved_insight:
        st.info(
            "The selected month or comparison changed. "
            "Generate a new AI insight for the current dashboard."
        )


if __name__ == "__main__":
    main()
