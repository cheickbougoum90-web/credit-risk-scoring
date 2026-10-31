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

- **Artefact de données (fuite de la cible) :** dans le jeu d'entraînement, `loan_percent_income` fourni diffère de `loan_amnt / person_income` sur 211 lignes (1,1 %), et **toutes sont des défauts**. Le modèle utilise donc le ratio recalculé `loan_to_income` à la place (voir plus bas).

![Taux de défaut par variable numérique](reports/figures/eda_default_rate_numeric.png)

## Décisions de nettoyage

Décidées à partir de l'analyse du jeu d'entraînement ; la règle déterministe est dans [`src/data/clean.py`](src/data/clean.py), et les imputations sont apprises dans le pipeline de prétraitement pour qu'aucune information de la validation ou du test ne fuite.

| Sujet | Décision | Implémentée dans |
|---|---|---|
| Âge supérieur à 100 ans, ancienneté supérieure à 60 ans | Lignes supprimées avec la même règle sur l'entraînement, la validation et le test (6, 1 et 0 lignes). L'API refuse ces valeurs. | `src/data/clean.py` |
| Revenu supérieur à 1 000 000 | Conservé ; transformation logarithmique ou mise à l'échelle robuste pour les modèles linéaires | pipeline de prétraitement |
| `person_emp_length` manquant | Imputation par la médiane plus un indicateur de valeur manquante (31,2 % de défauts quand elle manque contre 21,6 %) | pipeline de prétraitement |
| `loan_int_rate` manquant | Imputation par la médiane du `loan_grade`, calculée sur l'entraînement | pipeline de prétraitement |
| `loan_percent_income` | Remplacé par le ratio recalculé `loan_to_income` : le ratio fourni diffère sur 1,1 % des lignes d'entraînement, toutes des défauts (artefact de fuite du jeu de données). Ces lignes sont conservées, pas supprimées. | `src/features/engineering.py` |
| `loan_grade` | Conservé ; comparé avec et sans | étude dédiée |

## Comparaison des modèles

Métriques sur le **jeu de validation** (un seul découpage, pas encore d'intervalle de confiance). Le rappel et la précision sont donnés au seuil de 0,5 uniquement à titre de référence ; le seuil optimal en coût est choisi plus tard. Tableau complet : [`reports/model_comparison.csv`](reports/model_comparison.csv).

| Modèle | ROC AUC | PR AUC | KS | Brier | Rappel | Précision |
|---|---|---|---|---|---|---|
| prior constant (sans modèle) | 0,500 | 0,219 | 0,000 | 0,171 | 0,000 | 0,000 |
| régression logistique (baseline) | 0,877 | 0,723 | 0,611 | 0,104 | 0,544 | 0,741 |
| logistique + `loan_to_income` (remplace `loan_percent_income`) | 0,878 | 0,726 | 0,609 | 0,103 | 0,550 | 0,744 |
| LightGBM (réglages par défaut) | 0,946 | 0,905 | 0,759 | 0,051 | 0,752 | 0,954 |
| LightGBM (réglé, non retenu) | 0,948 | 0,907 | 0,765 | 0,051 | 0,750 | 0,966 |

La sélection des variables a été validée par validation croisée sur le jeu d'entraînement uniquement (5 plis, régression logistique et gradient boosting). Sur quatre variables candidates, seul le ratio recalculé en remplacement du ratio fourni a été retenu ; l'ajouter à côté du ratio fourni aurait permis au modèle de lire la fuite décrite plus haut (détails dans [`notebooks/02_modeling_experiments.ipynb`](notebooks/02_modeling_experiments.ipynb), section 3). Les écarts entre les deux modèles logistiques sont faibles et viennent d'un seul découpage de validation.

![Courbes ROC et précision-rappel de la baseline](reports/figures/baseline_roc_pr.png)

### Validation croisée sur le jeu d'entraînement (issue #10)

Validation croisée stratifiée à 5 plis, moyenne ± écart-type entre les plis, mêmes plis pour tous les modèles, hyperparamètres proches des valeurs par défaut (ni réglage, ni pondération des classes) et variables sélectionnées ci-dessus. Tableau complet : [`reports/cv_comparison.csv`](reports/cv_comparison.csv).

| Modèle | PR AUC | ROC AUC | KS | Brier |
|---|---|---|---|---|
| XGBoost | 0,900 ± 0,004 | 0,946 ± 0,002 | 0,750 ± 0,008 | 0,054 ± 0,002 |
| LightGBM | 0,900 ± 0,004 | 0,944 ± 0,001 | 0,752 ± 0,012 | 0,054 ± 0,002 |
| Random Forest | 0,884 ± 0,005 | 0,932 ± 0,002 | 0,735 ± 0,015 | 0,059 ± 0,002 |
| arbre de décision (profondeur 5, diagnostic) | 0,824 ± 0,008 | 0,892 ± 0,005 | 0,679 ± 0,015 | 0,067 ± 0,003 |
| régression logistique | 0,709 ± 0,018 | 0,867 ± 0,008 | 0,583 ± 0,017 | 0,107 ± 0,003 |

![Comparaison des modèles](reports/figures/model_cv_comparison.png)

XGBoost et LightGBM sont équivalents (écart apparié de PR AUC +0,0002, écart-type 0,0018) ; **LightGBM a été choisi** car il s'entraîne un peu plus vite, et XGBoost reste une alternative à revoir au moment du réglage. Les deux dépassent la régression logistique sur chaque pli. Relancer les modèles à arbres avec les variables d'origine (ratio fuyant inclus) n'a rien changé de significatif : les arbres n'exploitent pas cet artefact. Ces scores sont très élevés pour du scoring de crédit et reflètent probablement la construction de ce jeu de données public (voir Limites). La ligne LightGBM du tableau de validation ci-dessus est une évaluation unique sur le jeu de validation ; rappel et précision sont donnés au seuil de référence 0,5.

### Déséquilibre des classes (issue #11)

Les poids de classes (`balanced`) et SMOTE-NC (1:1, appliqué uniquement à la partie entraînement de chaque pli) ont été comparés aux modèles non pondérés, par validation croisée à 5 plis sur le jeu d'entraînement. Tableau complet : [`reports/imbalance_comparison.csv`](reports/imbalance_comparison.csv).

| Modèle | Stratégie | PR AUC | Brier | Probabilité moyenne prédite | Rappel à 0,5 | Précision à 0,5 |
|---|---|---|---|---|---|---|
| LightGBM | aucune | 0,900 | 0,054 | 0,208 | 0,731 | 0,957 |
| LightGBM | poids de classes | 0,897 | 0,061 | 0,260 | 0,771 | 0,862 |
| LightGBM | SMOTE-NC | 0,897 | 0,055 | 0,215 | 0,733 | 0,943 |
| Régression logistique | aucune | 0,709 | 0,107 | 0,219 | 0,517 | 0,747 |
| Régression logistique | poids de classes | 0,704 | 0,146 | 0,384 | 0,787 | 0,512 |
| Régression logistique | SMOTE-NC | 0,701 | 0,141 | 0,366 | 0,768 | 0,526 |

Le taux de défaut est de 21,9 %. Aucune des deux stratégies n'améliore la qualité du classement (la PR AUC est plus basse, sans qu'aucun pli ne s'améliore) et les deux dégradent les probabilités (score de Brier, probabilité moyenne prédite). Elles déplacent seulement le point de fonctionnement au seuil de 0,5 : un LightGBM non pondéré atteint le même rappel que la version pondérée (0,771) avec un seuil de 0,31 et une meilleure précision (0,874 contre 0,862). **Les modèles restent donc non pondérés** ; le déséquilibre est traité par la calibration des probabilités et par un seuil de décision fondé sur les coûts (issues suivantes). La pondération a aussi été proposée à la recherche d'hyperparamètres (voir plus bas) ; XGBoost n'a pas été testé.

### Réglage des hyperparamètres (issue #13)

Recherche Optuna pour le LightGBM retenu : 40 essais (échantillonneur TPE avec graine, premier essai proche de la configuration par défaut), objectif = PR AUC moyenne des mêmes 5 plis du jeu d'entraînement. Script : [`scripts/tune_lightgbm.py`](scripts/tune_lightgbm.py) ; résultats dans [`reports/tuning/`](reports/tuning/). Espace de recherche : nombre d'arbres (100 à 600), taux d'apprentissage (0,02 à 0,2), feuilles (8 à 96), minimum d'échantillons par feuille (5 à 80), sous-échantillonnage des lignes et des colonnes, régularisation L1/L2 et poids de classes.

| | PR AUC |
|---|---|
| LightGBM par défaut (validation croisée, mêmes plis) | 0,8998 |
| meilleur essai de la recherche (mêmes plis) | 0,9020 |
| gain apparié sur de nouveaux plis (graines 7 et 123) | +0,0021 et +0,0021 (bruit 0,0035 et 0,0038) |
| jeu de validation, par défaut / réglé | 0,9049 / 0,9071 |

Les 8 meilleurs essais forment un plateau (0,9006 à 0,9020), et le meilleur de 40 est biaisé à la hausse ; le gain a donc été mesuré de nouveau sur des plis que la recherche n'avait pas utilisés : +0,0021, inférieur à l'écart-type des différences appariées. Selon la règle d'adoption utilisée dans tout le projet, **le LightGBM par défaut est conservé** ; les paramètres réglés restent enregistrés dans [`reports/tuning/lightgbm_best_params.json`](reports/tuning/lightgbm_best_params.json). Les hyperparamètres comptent peu sur ce jeu de données. Le jeu de validation a été utilisé une seconde fois, pour ces deux modèles seulement.

![Historique de la recherche Optuna](reports/figures/tuning_history.png)

### Calibration des probabilités (issue #14)

La calibration sigmoïde (Platt) et isotonique a été comparée au LightGBM non calibré, par validation croisée à 5 plis sur le jeu d'entraînement ; le calibrateur est appris sur des prédictions hors pli du modèle de base, donc toute la procédure est évaluée sur des lignes qu'elle n'a jamais vues. Tableau complet : [`reports/calibration_comparison.csv`](reports/calibration_comparison.csv).

| Calibration | Brier | Perte logarithmique | ECE | Probabilité moyenne prédite | PR AUC |
|---|---|---|---|---|---|
| aucune (référence) | 0,0540 ± 0,0022 | 0,1955 ± 0,0049 | 0,0137 ± 0,0020 | 0,208 | 0,8998 ± 0,0043 |
| sigmoïde | 0,0540 ± 0,0020 | 0,1938 ± 0,0046 | 0,0109 ± 0,0023 | 0,222 | 0,8998 ± 0,0043 |
| isotonique | 0,0538 ± 0,0018 | 0,1926 ± 0,0061 | 0,0102 ± 0,0012 | 0,222 | 0,8926 ± 0,0048 |

Le taux de défaut est de 0,219. Le modèle non calibré est déjà raisonnablement calibré, et **aucun calibrateur n'améliore le score de Brier au-delà du bruit** ; la régression isotonique a l'ECE le plus bas mais fait baisser la PR AUC de 0,007 sur chaque pli. La sigmoïde améliore légèrement la perte logarithmique en validation croisée mais pas sur le jeu de validation, et elle corrige certaines plages de probabilité en en sur-corrigeant d'autres. **Aucun calibrateur n'est appliqué.** À garder en tête : le modèle sous-estime les risques faibles (environ 0,6 % prédit pour 1,5 % observé dans la tranche la plus basse), donc sa sortie se lit mieux comme un bon score de risque aux niveaux approximatifs.

![Courbes de fiabilité avant et après calibration](reports/figures/calibration_curves.png)

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
python -m venv .venv                # ou : conda create -n credit-risk python=3.12
source .venv/bin/activate           # Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q                 # aucun test n'a besoin du jeu de données
```

Pour reproduire les notebooks, télécharge le jeu de données dans `data/raw/` (voir plus haut) puis lance `notebooks/01_eda.ipynb`, ensuite `notebooks/02_modeling_experiments.ipynb`. `make install` et `make test` font la même chose sur les systèmes disposant de `make`.

## Limites

- **Le jeu de données semble synthétique.** Un artefact de construction est documenté plus haut (un ratio qui diffère uniquement sur les défauts), et les scores sont très élevés (ROC AUC autour de 0,95). Une telle performance n'est pas à attendre sur un portefeuille réel.
- **Pas d'information temporelle :** aucune validation hors période n'est possible. Les chiffres de validation viennent d'un seul découpage stratifié ; la validation croisée sur l'entraînement donne une dispersion entre plis, pas un contrôle externe.
- **Les hypothèses de coûts** (perte en cas de défaut de 60 %) sont des hypothèses ; le seuil de décision n'est pas encore optimisé.
- **Les probabilités sont raisonnablement calibrées mais pas exactes :** les risques faibles sont sous-estimés (environ 0,6 % prédit pour 1,5 % observé dans la tranche la plus basse). La calibration, la pondération des classes, SMOTE et le réglage des hyperparamètres ont été testés et non retenus. L'équité n'a pas encore été analysée.

## Licence

MIT
