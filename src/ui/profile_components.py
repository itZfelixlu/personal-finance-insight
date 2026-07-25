import streamlit as st


LIFE_STAGES = [
    "Prefer not to say",
    "Student",
    "Working professional",
    "Self-employed",
    "Retired",
    "Other",
]

HOUSEHOLDS = [
    "Prefer not to say",
    "Living alone",
    "With roommates",
    "With partner",
    "Family with children",
    "Other",
]

FINANCIAL_GOALS = [
    "No specific goal",
    "Reduce dining spending",
    "Control shopping",
    "Save for travel",
    "Build an emergency fund",
    "Keep spending stable",
]


def render_user_profile(selected_month):
    with st.expander("Personalize your AI insight (optional)"):
        st.caption(
            "This information is used only when you click "
            "Generate AI Insight. Avoid entering sensitive details."
        )

        life_stage = st.selectbox(
            "Life stage",
            LIFE_STAGES,
            key="profile_life_stage",
        )
        household = st.selectbox(
            "Household",
            HOUSEHOLDS,
            key="profile_household",
        )
        financial_goal = st.selectbox(
            "Main financial goal",
            FINANCIAL_GOALS,
            key="profile_financial_goal",
        )
        monthly_budget = st.number_input(
            "Monthly spending target (optional)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Leave this at $0 if you do not want to set a target.",
            key="profile_monthly_budget",
        )
        monthly_context = st.text_area(
            f"Anything unusual in {selected_month}? (optional)",
            placeholder=(
                "Examples: final exams, travel, moving, holidays, "
                "medical expenses, or starting a new job"
            ),
            max_chars=400,
            key=f"profile_monthly_context_{selected_month}",
        )

    return {
        "life_stage": (
            None if life_stage == "Prefer not to say" else life_stage
        ),
        "household": (
            None if household == "Prefer not to say" else household
        ),
        "financial_goal": (
            None if financial_goal == "No specific goal" else financial_goal
        ),
        "monthly_budget": monthly_budget if monthly_budget > 0 else None,
        "monthly_context": monthly_context.strip() or None,
    }


def profile_context_key(profile):
    return tuple(sorted(profile.items()))
