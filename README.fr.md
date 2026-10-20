# Credit Risk Scoring (scoring du risque de crédit)

Projet de scoring du risque de crédit de bout en bout : analyse exploratoire, modélisation, calibration des probabilités, interprétabilité avec SHAP, analyse d'équité et déploiement avec FastAPI et Docker.

> 🇬🇧 English version: [README.md](README.md)

**Statut :** 🚧 en cours. L'avancement est suivi dans les [Issues](../../issues) et les [Milestones](../../milestones).

## Problème métier

_À compléter (issue : « Define business problem and cost matrix »)._
Un prêteur doit décider d'accorder ou non un prêt. Accorder un prêt qui finit en défaut coûte beaucoup plus cher que refuser un bon client. L'objectif est de prédire la probabilité de défaut et de choisir un seuil de décision qui minimise le coût attendu.

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
