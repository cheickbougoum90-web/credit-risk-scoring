import pytest

pytest.importorskip("optuna")
pytest.importorskip("lightgbm")

from src.features.preprocessing import get_xy
from src.models.tuning import (
    FIXED,
    SPACE,
    build_tuned_model,
    load_best_params,
    run_study,
    save_results,
    trials_frame,
)
from tests.test_baseline import synthetic


@pytest.fixture(scope="module")
def study():
    X, y = get_xy(synthetic(n=600, seed=3))
    return run_study(X, y, n_trials=3, seed=1, n_splits=3), (X, y)


def test_trials_stay_inside_the_documented_search_space(study):
    s, _ = study
    assert len(s.trials) == 3
    for t in s.trials:
        for name, spec in SPACE.items():
            v = t.params[name]
            if spec[0] == "categorical":
                assert v in spec[1]
            else:
                assert spec[1] <= v <= spec[2], (name, v)


def test_first_trial_is_the_default_like_starting_point(study):
    s, _ = study
    assert s.trials[0].params["num_leaves"] == 31 and s.trials[0].params["class_weight"] is None


def test_objective_values_are_valid_pr_auc(study):
    s, _ = study
    assert all(0.0 < t.value <= 1.0 for t in s.trials)
    assert s.best_value == max(t.value for t in s.trials)


def test_same_seed_gives_the_same_search(study):
    s, (X, y) = study
    again = run_study(X, y, n_trials=3, seed=1, n_splits=3)
    assert [t.params for t in again.trials] == [t.params for t in s.trials]
    assert [t.value for t in again.trials] == pytest.approx([t.value for t in s.trials])


def test_tuned_model_receives_the_parameters_and_the_fixed_ones(study):
    s, (X, y) = study
    model = build_tuned_model(s.best_params)
    lgbm = model.named_steps["model"]
    assert lgbm.get_params()["num_leaves"] == s.best_params["num_leaves"]
    assert lgbm.get_params()["subsample_freq"] == FIXED["subsample_freq"]
    p = model.fit(X, y).predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_results_roundtrip_to_disk_keeps_none_class_weight(study, tmp_path):
    s, _ = study
    save_results(s, tmp_path)
    assert load_best_params(tmp_path / "lightgbm_best_params.json") == s.best_params
    frame = trials_frame(s)
    assert len(frame) == 3 and {"trial", "pr_auc", "pr_auc_std", "num_leaves"} <= set(frame.columns)
