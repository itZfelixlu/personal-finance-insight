import plotly.express as px
import streamlit as st


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
