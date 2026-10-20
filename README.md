# Credit Risk Scoring

End-to-end credit risk scoring project: exploratory analysis, modeling, probability calibration, SHAP explainability, fairness check, and deployment with FastAPI and Docker.

> 🇫🇷 Version française : [README.fr.md](README.fr.md)

**Status:** 🚧 work in progress. Progress is tracked in the [Issues](../../issues) and [Milestones](../../milestones).

## Business problem

_To be completed (issue: "Define business problem and cost matrix")._
A lender must decide whether to grant a loan. Granting a loan that ends in default costs far more than refusing a good customer. The goal is to predict the probability of default and choose a decision threshold that minimizes expected cost.

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
