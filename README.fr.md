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

## Résultats clés

_À compléter en fin de projet (AUC-ROC, AUC-PR, KS, impact financier estimé)._

## Structure du projet

```
credit-risk-scoring/
├── data/            # données brutes et transformées (non versionnées)
├── docs/            # dictionnaire des variables, model card
├── models/          # modèles sérialisés (non versionnés)
├── notebooks/       # EDA et expérimentations
├── reports/figures/ # figures utilisées dans le README
├── scripts/         # scripts utilitaires (inspection des données, etc.)
├── src/
│   ├── data/        # chargement et nettoyage
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
