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

## Key results

_To be completed at the end of the project (AUC-ROC, AUC-PR, KS, estimated financial impact)._

## Project structure

```
credit-risk-scoring/
├── data/            # raw and processed data (not versioned)
├── docs/            # data dictionary, model card
├── models/          # serialized models (not versioned)
├── notebooks/       # EDA and experiments
├── reports/figures/ # figures used in the README
├── scripts/         # helper scripts (data inspection, etc.)
├── src/
│   ├── data/        # loading and cleaning
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
