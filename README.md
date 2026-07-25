# Personal Finance Insight

A Streamlit application that analyzes Chase transaction CSV files. It
categorizes transactions, detects weekly spending patterns, compares monthly
spending, and can generate personalized AI insights.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run the App

From the project root:

```bash
streamlit run app.py
```

Upload one or more Chase CSV statements containing:

- `Posting Date`, `Date`, or `Transaction Date`
- `Description`
- `Amount`

Multiple monthly statements can be uploaded together for month-to-month
comparison.

## Optional AI Insights

The dashboard works without an OpenAI API key. An API key is required only for
the **Generate AI Insight** feature.

Create this local file:

```text
.streamlit/secrets.toml
```

Add your key:

```toml
OPENAI_API_KEY = "your-api-key"
```

An optional model can also be configured:

```toml
OPENAI_MODEL = "gpt-5.6-luna"
```

API keys are sensitive. Do not place a key directly in `app.py` or commit
`.streamlit/secrets.toml` to GitHub. The secrets file is already excluded by
`.gitignore`.

## Test Statements

Sample single-month statements are available in `test_inputs/`. You can upload
one file to test single-month analysis or several files to test monthly
comparison.
