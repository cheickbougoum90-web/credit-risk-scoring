# Data dictionary

Source: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle).
Shape: **32,581 rows x 12 columns**. Target: `loan_status` (1 = default, 21.8% of rows).

Types, missing counts, ranges and categories below were computed from the raw file.
Column meanings are inferred from the column names and should be checked against the Kaggle page.

| Column | Type | Missing | Range / categories | Meaning (to verify) |
|---|---|---|---|---|
| person_age | int | 0 | 20 to 144 | Age of the applicant (years) |
| person_income | int | 0 | 4,000 to 6,000,000 | Annual income |
| person_home_ownership | categorical | 0 | RENT, MORTGAGE, OWN, OTHER | Home ownership status |
| person_emp_length | float | 895 (2.7%) | 0 to 123 | Employment length (years) |
| loan_intent | categorical | 0 | EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT | Purpose of the loan |
| loan_grade | categorical | 0 | A, B, C, D, E, F, G | Loan grade assigned by the lender |
| loan_amnt | int | 0 | 500 to 35,000 | Loan amount |
| loan_int_rate | float | 3,116 (9.6%) | 5.42 to 23.22 | Interest rate (%) |
| loan_status | binary | 0 | 0 = no default, 1 = default | **Target** |
| loan_percent_income | float | 0 | 0.00 to 0.83 | Loan amount / income (matches `loan_amnt / person_income` on 99% of rows) |
| cb_person_default_on_file | categorical | 0 | Y, N | Past default recorded in the credit bureau file |
| cb_person_cred_hist_length | int | 0 | 2 to 30 | Credit history length (years) |

## Data quality findings (raw data)

- **Duplicated rows:** 165.
- **Implausible ages:** 5 applicants above 100 years (maximum 144).
- **Implausible employment length:** 2 rows above 60 years (maximum 123). Two rows also have an employment length larger than age minus 14.
- **Extreme incomes:** 9 rows above 1,000,000 (maximum 6,000,000), to be examined before any scaling.
- **Missing values:** `loan_int_rate` (3,116) and `person_emp_length` (895). Missingness rate differs slightly between classes for `person_emp_length` (2.4% for non-defaults, 4.0% for defaults), so missingness may carry signal.
- **Class balance:** 78.2% no default, 21.8% default. A model that always predicts "no default" reaches 78.2% accuracy.
- **Default rate by `loan_grade`:** A 10.0%, B 16.3%, C 20.7%, D 59.0%, E 64.4%, F 70.5%, G 98.4%. The grade already summarizes a risk assessment (see the dedicated issue on its impact).
