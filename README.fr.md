# Credit Risk Scoring (scoring du risque de crédit)

Projet de scoring du risque de crédit de bout en bout : analyse exploratoire, modélisation, calibration des probabilités, interprétabilité avec SHAP, analyse d'équité et déploiement avec FastAPI et Docker.

> 🇬🇧 English version: [README.md](README.md)

**Statut :** 🚧 en cours. L'avancement est suivi dans les [Issues](../../issues) et les [Milestones](../../milestones).

## Problème métier

Un prêteur doit décider d'accorder ou non un prêt. Deux erreurs sont possibles :

- **Faux négatif** (prêt accordé, l'emprunteur fait défaut) : le prêteur perd une partie du capital. Perte en cas de défaut (LGD) supposée : **60 % du montant du prêt**.
- **Faux positif** (bon client refusé) : le prêteur perd les intérêts attendus, approximés par **loan_amnt × loan_int_rate** (approximation sur un an).

Avec le taux d'intérêt moyen du jeu de données, un défaut non détecté coûte environ **5 fois** plus qu'un bon client refusé à tort. Ces hypothèses étant discutables, les résultats sont aussi présentés pour d'autres ratios de coûts (3:1, 5:1, 10:1).

**Objectif :** prédire la probabilité de défaut, puis choisir le seuil de décision qui minimise le coût total attendu.

**Métriques :** AUC-PR et KS (qualité du classement), score de Brier (calibration), coût attendu (impact métier). L'accuracy n'est pas utilisée : un modèle naïf qui prédit toujours « pas de défaut » atteint déjà environ 78 %.

## Jeu de données

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle). Les données brutes ne sont pas versionnées : télécharge le CSV et place-le dans `data/raw/`.

## Enseignements sur les données

Analyse exploratoire sur le jeu d'entraînement uniquement (notebook complet : [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)). Le jeu contient 32 581 lignes ; 165 doublons exacts ont été supprimés avant le découpage stratifié 60/20/20 (21,9 % de défauts dans chaque partie).

- **Signaux forts et non linéaires :** `loan_grade` (10 % de défauts en grade A, 60 % en grade D), `loan_int_rate`, `loan_percent_income` (forte hausse au-delà de 0,25) et `person_income`.
- **Le grade et le taux d'intérêt se recoupent fortement** (taux médian de 7,5 % en grade A, 20,1 % en grade G), d'où une étude dédiée avec et sans `loan_grade`.
- **Qualité des données :** valeurs invraisemblables (âges jusqu'à 144 ans, ancienneté jusqu'à 123 ans), 9,5 % de `loan_int_rate` et 2,8 % de `person_emp_length` manquants. Une ancienneté manquante va avec un taux de défaut plus élevé (31,2 % contre 21,6 %).

![Taux de défaut par variable numérique](reports/figures/eda_default_rate_numeric.png)

## Décisions de nettoyage

Décidées à partir de l'analyse du jeu d'entraînement ; la règle déterministe est dans [`src/data/clean.py`](src/data/clean.py), et les imputations sont apprises dans le pipeline de prétraitement pour qu'aucune information de la validation ou du test ne fuite.

| Sujet | Décision | Implémentée dans |
|---|---|---|
| Âge supérieur à 100 ans, ancienneté supérieure à 60 ans | Lignes supprimées avec la même règle sur l'entraînement, la validation et le test (6, 1 et 0 lignes). L'API refuse ces valeurs. | `src/data/clean.py` |
| Revenu supérieur à 1 000 000 | Conservé ; transformation logarithmique ou mise à l'échelle robuste pour les modèles linéaires | pipeline de prétraitement |
| `person_emp_length` manquant | Imputation par la médiane plus un indicateur de valeur manquante (31,2 % de défauts quand elle manque contre 21,6 %) | pipeline de prétraitement |
| `loan_int_rate` manquant | Imputation par la médiane du `loan_grade`, calculée sur l'entraînement | pipeline de prétraitement |
| `loan_percent_income` | Conservé tel quel ; le ratio recalculé est testé comme variable | feature engineering |
| `loan_grade` | Conservé ; comparé avec et sans | étude dédiée |

## Comparaison des modèles

Métriques sur le **jeu de validation** (un seul découpage, pas encore d'intervalle de confiance). Le rappel et la précision sont donnés au seuil de 0,5 uniquement à titre de référence ; le seuil optimal en coût est choisi plus tard. Tableau complet : [`reports/model_comparison.csv`](reports/model_comparison.csv).

| Modèle | ROC AUC | PR AUC | KS | Brier | Rappel | Précision |
|---|---|---|---|---|---|---|
| prior constant (sans modèle) | 0,500 | 0,219 | 0,000 | 0,171 | 0,000 | 0,000 |
| régression logistique (baseline) | 0,877 | 0,723 | 0,611 | 0,104 | 0,544 | 0,741 |

![Courbes ROC et précision-rappel de la baseline](reports/figures/baseline_roc_pr.png)

## Résultats clés

_À compléter en fin de projet (AUC-ROC, AUC-PR, KS, impact financier estimé)._

## Structure du projet

```
credit-risk-scoring/
├── data/            # données brutes et transformées (non versionnées)
├── docs/            # dictionnaire des variables, model card
├── models/          # modèles sérialisés (non versionnés)
├── notebooks/       # 01_eda, 02_modeling_experiments, 03_evaluation_explainability
├── reports/figures/ # figures utilisées dans le README
├── scripts/         # scripts utilitaires (inspection des données, etc.)
├── src/
│   ├── data/        # chargement, découpage et nettoyage
│   ├── features/    # prétraitement et feature engineering
│   ├── models/      # entraînement, évaluation, seuil
│   └── api/         # service FastAPI
└── tests/
```

## Démarrage

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
make install
make test
```

## Limites

_À compléter (par exemple : jeu de données probablement synthétique, pas d'information temporelle, pas de validation en conditions réelles)._

## Licence

MIT
