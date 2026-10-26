# Credit Risk Scoring

End-to-end credit risk scoring project: exploratory analysis, modeling, probability calibration, SHAP explainability, fairness check, and deployment with FastAPI and Docker.

> 🇫🇷 Version française : [README.fr.md](README.fr.md)

**Status:** 🚧 work in progress. Progress is tracked in the [Issues](../../issues) and [Milestones](../../milestones).

## Business problem

A lender must decide whether to grant a loan. Two errors are possible:

- **False negative** (loan granted, borrower defaults): the lender loses part of the principal. Assumed loss given default (LGD): **60% of the loan amount**.
- **False positive** (good customer refused): the lender loses the expected interest income, approximated by **loan_amnt × loan_int_rate** (one-year proxy).

With the average interest rate in the data, a missed default costs roughly **5 times** more than a wrongly refused customer. Because these assumptions are debatable, results are also reported for other cost ratios (3:1, 5:1, 10:1).

**Goal:** predict the probability of default, then choose the decision threshold that minimizes the expected total cost.

**Metrics:** AUC-PR and KS (ranking quality), Brier score (calibration), expected cost (business impact). Accuracy is not used: a naive model that always predicts "no default" already reaches about 78%.

## Dataset

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle). The raw data is not versioned: download it and place the CSV in `data/raw/`.

## Data insights

Exploratory analysis on the training set only (full notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)). The dataset has 32,581 rows; 165 exact duplicates were removed before the stratified 60/20/20 split (21.9% defaults in each part).

- **Strong, non-linear signals:** `loan_grade` (10% default in grade A, 60% in grade D), `loan_int_rate`, `loan_percent_income` (sharp rise above 0.25) and `person_income`.
- **Grade and interest rate overlap heavily** (median rate 7.5% in grade A, 20.1% in grade G), hence a dedicated with/without `loan_grade` study.
- **Data quality:** implausible values (ages up to 144, employment length up to 123 years), 9.5% of `loan_int_rate` and 2.8% of `person_emp_length` missing. Missing employment length goes with a higher default rate (31.2% versus 21.6%).

- **Data artifact (target leakage):** in the training set, the provided `loan_percent_income` differs from `loan_amnt / person_income` on 211 rows (1.1%), and **all of them are defaults**. The model therefore uses the recomputed ratio `loan_to_income` instead (see below).

![Default rate by numeric variable](reports/figures/eda_default_rate_numeric.png)

## Data cleaning decisions

Decided from the training-set analysis; the deterministic rule is in [`src/data/clean.py`](src/data/clean.py), the imputations are learned inside the preprocessing pipeline so that nothing leaks from validation or test data.

| Topic | Decision | Implemented in |
|---|---|---|
| Age above 100, employment length above 60 | Rows removed with the same rule on train, validation and test (6, 1 and 0 rows). The API rejects such inputs. | `src/data/clean.py` |
| Income above 1,000,000 | Kept; log-transform or robust scaling for linear models | preprocessing pipeline |
| Missing `person_emp_length` | Median imputation plus a missing-value indicator (31.2% default rate when missing versus 21.6%) | preprocessing pipeline |
| Missing `loan_int_rate` | Median imputation by `loan_grade`, fitted on train | preprocessing pipeline |
| `loan_percent_income` | Replaced by the recomputed ratio `loan_to_income`: the provided ratio differs on 1.1% of training rows, all defaults (a leakage artifact of the dataset). Those rows are kept, not deleted. | `src/features/engineering.py` |
| `loan_grade` | Kept; compared with and without | dedicated study |

## Model comparison

Metrics on the **validation set** (single split, no confidence interval yet). Recall and precision are given at a threshold of 0.5 only as a reference; the cost-optimal threshold is chosen later. Full table: [`reports/model_comparison.csv`](reports/model_comparison.csv).

| Model | ROC AUC | PR AUC | KS | Brier | Recall | Precision |
|---|---|---|---|---|---|---|
| constant prior (no model) | 0.500 | 0.219 | 0.000 | 0.171 | 0.000 | 0.000 |
| logistic regression (baseline) | 0.877 | 0.723 | 0.611 | 0.104 | 0.544 | 0.741 |
| logistic + `loan_to_income` (replaces `loan_percent_income`) | 0.878 | 0.726 | 0.609 | 0.103 | 0.550 | 0.744 |

Feature selection was cross-validated on the training set only (5 folds, logistic regression and gradient boosting). Of four candidate features, only the recomputed ratio replacing the provided one was kept; adding it next to the provided ratio would have let the model read the leak above (details in [`notebooks/02_modeling_experiments.ipynb`](notebooks/02_modeling_experiments.ipynb), section 3). The differences between the two logistic models are small and come from a single validation split.

![Baseline ROC and precision-recall curves](reports/figures/baseline_roc_pr.png)

## Key results

_To be completed at the end of the project (AUC-ROC, AUC-PR, KS, estimated financial impact)._

## Project structure

```
credit-risk-scoring/
├── data/            # raw and processed data (not versioned)
├── docs/            # data dictionary, model card
├── models/          # serialized models (not versioned)
├── notebooks/       # 01_eda, 02_modeling_experiments, 03_evaluation_explainability
├── reports/figures/ # figures used in the README
├── scripts/         # helper scripts (data inspection, etc.)
├── src/
│   ├── data/        # loading, splitting and cleaning
│   ├── features/    # preprocessing and feature engineering
│   ├── models/      # training, evaluation, threshold
│   └── api/         # FastAPI service
└── tests/
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
make install
make test
```

## Limitations

_To be completed (for example: dataset likely synthetic, no temporal information, no real-world validation)._

## License

MIT
