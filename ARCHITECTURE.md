# Personal Finance Insight Tool

## Overview

This project builds a personal finance insight tool that automatically analyzes Chase bank transaction CSV files.

The system consists of two machine learning components:

1. **Supervised Learning**
   - Predicts transaction categories for uncategorized bank transactions.

2. **Unsupervised Learning**
   - Discovers weekly spending patterns based on categorized transactions.

Finally, the system generates personalized financial insights through a Streamlit dashboard.

---

# Overall Pipeline

```
User uploads Chase CSV
        │
        ▼
Data Cleaning
        │
        ▼
Supervised Learning
(Transaction Classification)
        │
        ▼
Transaction DataFrame
(+ Category Column)
        │
        ▼
Weekly Aggregation
        │
        ▼
Weekly Feature Engineering
        │
        ▼
Unsupervised Learning
(Weekly Spending Pattern Discovery)
        │
        ▼
Insight Generation
        │
        ▼
Dashboard
```

---

# Project Structure

```
personal-finance-insight/

│
├── README.md
├── requirements.txt
├── app.py
│
├── inputs/
│   ├── synthetic_transactions.csv
│   └── sample_chase.csv
│
├── notebooks/
│   └── experiments.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── categorizer.py
│   ├── weekly_features.py
│   ├── pattern_predictor.py
│   ├── insight_generator.py
│   └── visualization.py
│
├── models/
│   ├── category_model.pkl
│   ├── weekly_scaler.pkl
│   ├── spending_pattern_model.pkl
│   ├── cluster_names.json
│   └── feature_config.json
│
└── outputs/
    ├── categorized_transactions.csv
    └── weekly_summary.csv
```

---

# Folder Responsibilities

## inputs/

Contains all input datasets.

Examples:

- synthetic_transactions.csv (training dataset)
- sample_chase.csv (example inference dataset)

This folder stores datasets only.

---

## notebooks/

Contains all machine learning experiments.

Responsibilities:

- Load training dataset from `inputs/`
- Perform preprocessing
- Train classification models
- Evaluate model performance
- Train clustering model
- Export trained models to `models/`

The notebook **does not generate datasets**.

---

## src/

Contains all production Python modules.

Every module should have a single responsibility.

---

### data_loader.py

#### Responsibilities

- Read uploaded Chase CSV
- Validate required columns

#### Input

```text
Uploaded Chase CSV
```

#### Output

```text
Raw Transaction DataFrame

Date
Description
Amount
```

#### Notes

The output becomes the input to `preprocessing.py`.

---

### preprocessing.py

#### Responsibilities

- Clean transaction data
- Convert dates
- Normalize merchant names if necessary
- Remove invalid records
- Prepare the DataFrame for downstream tasks

#### Input

```text
Raw Transaction DataFrame

Date
Description
Amount
```

#### Output

```text
Clean Transaction DataFrame

Date
Description
Amount
```

#### Notes

This module must be shared between both the training notebook and the production application.

The output becomes the input to `categorizer.py`.

---

### categorizer.py

#### Responsibilities

- Load `category_model.pkl`
- Predict transaction categories
- Add a **Category** column

#### Input

```text
Clean Transaction DataFrame

Date
Description
Amount
```

#### Output

```text
Categorized Transaction DataFrame

Date
Description
Amount
Category
```

#### Notes

The output becomes the input to `weekly_features.py`.

---

### weekly_features.py

#### Responsibilities

Aggregate categorized transactions into weekly feature vectors.

Typical features include:

- Dining Ratio
- Shopping Ratio
- Grocery Ratio
- Transportation Ratio
- Total Spending
- Transaction Count
- Average Transaction

#### Input

```text
Categorized Transaction DataFrame

Date
Description
Amount
Category
```

#### Output

```text
Weekly Feature DataFrame

Week
Feature 1
Feature 2
...
Feature N
```

#### Notes

This module must be shared between training and inference.

The output becomes the input to `pattern_predictor.py`.

---

### pattern_predictor.py

#### Responsibilities

- Load `weekly_scaler.pkl`
- Load `spending_pattern_model.pkl`
- Predict weekly spending patterns

#### Input

```text
Weekly Feature DataFrame
```

#### Output

```text
Weekly Pattern DataFrame

Week
Pattern ID
Pattern Name
```

#### Notes

The output becomes the input to `insight_generator.py`.

---

### insight_generator.py

#### Responsibilities

Generate natural language insights from machine learning outputs.

#### Input

```text
Weekly Pattern DataFrame

Week
Pattern Name
```

#### Output

```text
Natural Language Insights
```

Example

```text
Dining spending increased this week.

Transportation spending remains stable.

Overall spending pattern resembles a Dining-heavy week.
```

#### Notes

This module should only consume machine learning outputs.

It must not modify predictions or retrain models.

---

### visualization.py

#### Responsibilities

Generate visualizations for the Streamlit dashboard.

#### Input

```text
Weekly Pattern DataFrame

Natural Language Insights
```

#### Output

```text
Interactive Dashboard
```

Typical visualizations include:

- Spending trend
- Category breakdown
- Weekly summary

#### Notes

This module is responsible only for presentation and should not contain machine learning logic.

---

# Machine Learning Responsibilities

## Supervised Learning

### Purpose

Predict a transaction category for each uncategorized transaction.

### Input

Transaction DataFrame

```
Date | Description | Amount
```

### Output

Transaction DataFrame

```
Date | Description | Amount | Category
```

### Example

Before

```
2025-01-03 | STARBUCKS | -6.25
```

↓

After

```
2025-01-03 | STARBUCKS | -6.25 | Dining
```

The output of the supervised learning stage becomes the input to the weekly aggregation stage.

---

## Unsupervised Learning

Purpose

Discover weekly spending behavior.

Input

```
Weekly Feature Vector
```

Output

```
Pattern ID
```

Example

```
Dining-heavy

Essential-focused

Balanced Spending

High Spending
```

This model **should NOT classify transactions**.

---

# Experiment Pipeline

```
Load Training Dataset
(from inputs/)
        │
        ▼
Preprocessing
        │
        ▼
TF-IDF
        │
        ▼
Train Classifier
        │
        ▼
Model Evaluation
        │
        ▼
Save category_model.pkl
        │
        ▼
Weekly Aggregation
        │
        ▼
Feature Engineering
        │
        ▼
StandardScaler
        │
        ▼
Train KMeans
        │
        ▼
Export Models
```

---

# Product Pipeline

```
Upload Chase CSV
        │
        ▼
Load CSV
        │
        ▼
Preprocessing
        │
        ▼
Category Prediction
(Add Category Column)
        │
        ▼
Weekly Aggregation
        │
        ▼
Weekly Feature Engineering
        │
        ▼
Scaler
        │
        ▼
KMeans Prediction
        │
        ▼
Insight Generation
        │
        ▼
Dashboard
```

---

# Data Flow

```
Raw Transaction DataFrame
(Date, Description, Amount)
            │
            ▼
preprocessing.py
            │
            ▼
Clean Transaction DataFrame
            │
            ▼
categorizer.py
            │
            ▼
Categorized Transaction DataFrame
(+ Category)
            │
            ▼
weekly_features.py
            │
            ▼
Weekly Feature DataFrame
            │
            ▼
pattern_predictor.py
            │
            ▼
Weekly Pattern DataFrame
(+ Pattern Name)
            │
            ▼
insight_generator.py
            │
            ▼
Natural Language Insights
            │
            ▼
visualization.py
            │
            ▼
Streamlit Dashboard
```

---

# Model Files

## category_model.pkl

Predict transaction category.

---

## weekly_scaler.pkl

Normalize weekly feature vectors before clustering.

---

## spending_pattern_model.pkl

Predict weekly spending patterns.

---

## cluster_names.json

Maps cluster IDs into readable pattern names.

Example

```
0 → Dining-heavy
1 → Essential-focused
2 → Balanced Spending
3 → High Spending
```

---

## feature_config.json

Stores the feature ordering used during training.

The production application must always use the same feature order.

---

# Development Guidelines

## General Principles

1. Each Python module should have a single responsibility.

2. The experiment notebook is responsible only for model training, evaluation, and exporting trained models.

3. The production application (`app.py`) must never train machine learning models. It only loads trained models from the `models/` directory.

---

## Code Organization

4. All transaction preprocessing must be implemented in `src/preprocessing.py`.

   Both the experiment notebook and the production application must import and use the same preprocessing functions.

5. All weekly feature engineering must be implemented in `src/weekly_features.py`.

   Both the experiment notebook and the production application must use the same feature engineering functions.

6. Do not implement preprocessing or feature engineering directly inside `app.py` or the notebook unless it is for temporary experimentation.

---

## Project Structure

7. All training datasets should be placed inside the `inputs/` directory.

8. All trained models should be exported to the `models/` directory.

9. All generated results should be written to the `outputs/` directory.

---

## Machine Learning Responsibilities

10. Supervised Learning is responsible only for transaction classification.

11. Unsupervised Learning is responsible only for discovering weekly spending patterns.

12. Insight generation should only consume the outputs of the machine learning models. It should not modify predictions or retrain models.