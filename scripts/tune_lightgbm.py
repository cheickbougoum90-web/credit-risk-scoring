"""Run the Optuna study for the chosen LightGBM model (issue #13).

    python -m scripts.tune_lightgbm            # 40 trials, seed 42
    python -m scripts.tune_lightgbm --trials 60

Uses the cleaned TRAINING set only. Writes reports/tuning/lightgbm_trials.csv and
reports/tuning/lightgbm_best_params.json. The notebook only reads these files.
"""
import argparse
import time

from src.data.clean import load_clean_split
from src.data.load import PROJECT_ROOT
from src.features.preprocessing import get_xy
from src.models.tuning import run_study, save_results

OUT_DIR = PROJECT_ROOT / "reports" / "tuning"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = get_xy(load_clean_split("train"))
    start = time.perf_counter()
    study = run_study(X, y, n_trials=args.trials, seed=args.seed)
    save_results(study, OUT_DIR)
    first = study.trials[0].value
    print(f"{len(study.trials)} trials in {time.perf_counter() - start:.0f}s")
    print(f"start point (close to the default): PR AUC {first:.4f}")
    print(f"best (trial {study.best_trial.number}): PR AUC {study.best_value:.4f}")
    print(f"results saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
