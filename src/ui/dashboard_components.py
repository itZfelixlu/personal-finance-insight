import plotly.express as px
import streamlit as st


def render_spending_chart(category_spending, selected_month):
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


def render_month_comparison(
    comparison,
    selected_month,
    comparison_month,
):
    """Render month-over-month spending comparison."""
    if comparison is None:
        return

    st.divider()
    st.subheader(
        f"{selected_month} compared to {comparison_month}"
    )

    total_change = comparison["total_change"]
    change_percent = comparison["total_change_percent"]

    if change_percent is None:
        delta_text = "No comparison spending"
    else:
        delta_text = f"{change_percent:+.1f}%"

    st.metric(
        f"Spending change compared to {comparison_month}",
        f"${total_change:+,.2f}",
        delta=delta_text,
        delta_color="inverse",
    )

    with st.expander(
        "View category comparison",
        expanded=True,
    ):
        st.dataframe(
            comparison["category_comparison"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Current": st.column_config.NumberColumn(
                    format="$%.2f"
                ),
                "Comparison": st.column_config.NumberColumn(
                    format="$%.2f"
                ),
                "Change": st.column_config.NumberColumn(
                    format="$%.2f"
                ),
                "Change %": st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
            },
        )


def _render_pattern_table(weekly_patterns):
    """Render a compact, user-facing view of weekly model predictions."""
    pattern_table = weekly_patterns[
        [
            "AnalysisPeriod",
            "PatternName",
            "total_spend",
            "txn_count",
        ]
    ].copy()
    pattern_table = pattern_table.rename(
        columns={
            "AnalysisPeriod": "Analyzed period",
            "PatternName": "Detected pattern",
            "total_spend": "Period spending",
            "txn_count": "Transactions",
        }
    )

    st.dataframe(
        pattern_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Period spending": st.column_config.NumberColumn(
                format="$%.2f"
            ),
            "Transactions": st.column_config.NumberColumn(format="%d"),
        },
    )


def render_weekly_patterns(
    weekly_patterns,
    selected_month,
    comparison_weekly_patterns=None,
    comparison_month=None,
):
    """Show the KMeans weekly-pattern outputs used by the AI insight."""
    st.markdown("#### Detected weekly spending patterns")
    st.caption(
        "These patterns are predicted by the unsupervised model and "
        "used as the main evidence for the AI explanation. Month-boundary "
        "rows are labeled as partial weeks and include only the displayed "
        "month's transactions. Treat partial-week patterns as "
        "lower-confidence signals."
    )

    if weekly_patterns.empty:
        st.info(f"No weekly spending pattern was found for {selected_month}.")
        return

    if comparison_weekly_patterns is None:
        _render_pattern_table(weekly_patterns)
        return

    current_tab, comparison_tab = st.tabs(
        [selected_month, comparison_month]
    )
    with current_tab:
        _render_pattern_table(weekly_patterns)
    with comparison_tab:
        if comparison_weekly_patterns.empty:
            st.info(
                f"No weekly spending pattern was found for "
                f"{comparison_month}."
            )
        else:
            _render_pattern_table(comparison_weekly_patterns)
