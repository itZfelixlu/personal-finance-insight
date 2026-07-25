import pandas as pd


def _prepare_month_data(transactions, selected_month):
    monthly_df = transactions[
        transactions["Month"] == selected_month
    ].copy()

    expenses_df = monthly_df[
        (monthly_df["Amount"] < 0)
        & (monthly_df["Category"] != "Transfer")
    ].copy()
    expenses_df["Spend"] = -expenses_df["Amount"]

    return monthly_df, expenses_df


def _summarize_spending(expenses):
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


def analyze_month(transactions, month):
    monthly_df, expenses_df = _prepare_month_data(
        transactions,
        month,
    )

    return {
        "transactions": monthly_df,
        "expenses": expenses_df,
        "summary": _summarize_spending(expenses_df),
    }


def compare_spending_summaries(current_summary, comparison_summary):
    current_categories = (
        current_summary["category_spending"]
        [["Category", "Spend"]]
        .rename(columns={"Spend": "Current"})
    )

    comparison_categories = (
        comparison_summary["category_spending"]
        [["Category", "Spend"]]
        .rename(columns={"Spend": "Comparison"})
    )

    category_comparison = pd.merge(
        current_categories,
        comparison_categories,
        on="Category",
        how="outer",
    ).fillna(0)

    category_comparison["Change"] = (
        category_comparison["Current"]
        - category_comparison["Comparison"]
    )

    category_comparison["Change %"] = (
        category_comparison["Change"]
        .div(
            category_comparison["Comparison"].replace(0, pd.NA)
        )
        * 100
    )

    category_comparison = category_comparison.sort_values(
        "Change",
        ascending=False,
    ).reset_index(drop=True)

    current_total = current_summary["total_spending"]
    comparison_total = comparison_summary["total_spending"]
    total_change = current_total - comparison_total

    total_change_percent = (
        total_change / comparison_total * 100
        if comparison_total != 0
        else None
    )

    return {
        "current_total": current_total,
        "comparison_total": comparison_total,
        "total_change": total_change,
        "total_change_percent": total_change_percent,
        "category_comparison": category_comparison,
    }
