import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    classification_metrics,
    comparison_to_markdown,
    ks_statistic,
    update_comparison_table,
)

Y = [0, 0, 1, 1]
S = [0.1, 0.4, 0.35, 0.8]  # textbook example: ROC AUC = 0.75


def test_known_values_on_textbook_example():
    m = classification_metrics(Y, S, threshold=0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["pr_auc"] == pytest.approx(0.8333, abs=1e-4)
    assert m["ks"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(0.158125)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 1, 1)
    assert m["recall"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(1.0)


def test_perfect_scores():
    m = classification_metrics([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0])
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["ks"] == 1.0


def test_constant_scores_carry_no_information():
    y = [0, 0, 0, 1]
    m = classification_metrics(y, [0.25] * 4)
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["ks"] == pytest.approx(0.0)
    assert m["pr_auc"] == pytest.approx(0.25)  # prevalence


def test_no_predicted_positive_gives_zero_precision_not_error():
    m = classification_metrics(Y, [0.1, 0.1, 0.1, 0.1], threshold=0.5)
    assert m["recall"] == 0.0 and m["precision"] == 0.0


def test_threshold_changes_thresholded_metrics_only():
    low = classification_metrics(Y, S, threshold=0.3)
    high = classification_metrics(Y, S, threshold=0.5)
    assert low["roc_auc"] == high["roc_auc"]
    assert low["recall"] > high["recall"]


def test_single_class_or_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ks_statistic([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0.1, 0.2, 0.3])


def test_comparison_table_create_replace_and_append(tmp_path):
    path = tmp_path / "reports" / "comparison.csv"
    m1 = classification_metrics(Y, S)
    update_comparison_table(path, "model_a", m1, notes="first")
    update_comparison_table(path, "model_b", m1)
    table = update_comparison_table(path, "model_a", m1, notes="updated")  # replaces model_a
    assert table["model"].tolist() == ["model_b", "model_a"]
    assert table.loc[table["model"] == "model_a", "notes"].iloc[0] == "updated"
    assert len(pd.read_csv(path)) == 2


def test_markdown_table_lists_only_requested_split(tmp_path):
    path = tmp_path / "c.csv"
    m = classification_metrics(Y, S)
    update_comparison_table(path, "model_a", m, split="validation")
    table = update_comparison_table(path, "model_a", m, split="test")
    md = comparison_to_markdown(table, split="validation")
    assert md.count("model_a") == 1
    assert md.splitlines()[0].startswith("| Model |")
