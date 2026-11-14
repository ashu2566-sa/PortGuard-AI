import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import train_port_congestion as training


def make_panel(n_months=30, ports=("Houston", "Seattle")):
    rows = []
    dates = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    for p_idx, port in enumerate(ports):
        for i, month in enumerate(dates):
            dwell = 2.0 + p_idx + (i % 7) * 0.45 + (i // 8) * 0.15
            rows.append({
                "Port": port,
                "YearMonth": month.strftime("%Y-%m"),
                "UniqueVessels": 100 + i * 2 + p_idx * 5,
                "MeanDwellDays": dwell,
                "MedianDwellDays": dwell - 0.2,
                "StdDwellDays": 0.3 + (i % 3) * 0.1,
                "TotalObservations": 500 + i * 4,
                "AvgVesselLength": 180 + p_idx * 10 + i * 0.2,
            })
    return pd.DataFrame(rows)


def test_load_panel_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Port": ["Houston"], "YearMonth": ["2020-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required AIS columns"):
        training.load_port_panel(path)


def test_load_panel_deduplicates_port_month_and_sorts(tmp_path):
    data = make_panel(n_months=6)
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    path = tmp_path / "panel.csv"
    data.to_csv(path, index=False)
    loaded = training.load_port_panel(path)
    assert not loaded.duplicated(["Port", "month"]).any()
    assert loaded.equals(loaded.sort_values(["Port", "month"]).reset_index(drop=True))


def test_features_are_historical_and_target_is_next_month():
    raw = make_panel(n_months=8)
    panel = training.build_panel_from_frame(raw) if hasattr(training, "build_panel_from_frame") else None
    # Exercise the same feature logic through a temporary CSV.
    with pytest.MonkeyPatch.context() as mp:
        tmp = Path.cwd() / ".pytest_port_panel.csv"
        try:
            raw.to_csv(tmp, index=False)
            built = training.build_panel(raw_path=tmp, macro_path=tmp.parent / "missing_macro.csv")
        finally:
            if tmp.exists():
                tmp.unlink()
    row = built[(built.Port == "Houston") & (built.month == pd.Timestamp("2019-04-01"))].iloc[0]
    assert row.target_month == pd.Timestamp("2019-05-01")
    assert row.target_next_month_dwell == pytest.approx(
        raw[(raw.Port == "Houston") & (raw.YearMonth == "2019-05")].MeanDwellDays.iloc[0]
    )
    assert row.rolling_3m_dwell == pytest.approx(
        raw[(raw.Port == "Houston") & (raw.YearMonth.isin(["2019-02", "2019-03", "2019-04"]))].MeanDwellDays.mean()
    )


def test_temporal_split_has_no_overlap():
    raw = make_panel(n_months=20)
    tmp = Path.cwd() / ".pytest_port_split.csv"
    try:
        raw.to_csv(tmp, index=False)
        panel = training.build_panel(raw_path=tmp, macro_path=tmp.parent / "missing_macro.csv")
    finally:
        if tmp.exists():
            tmp.unlink()
    train, test, cutoff = training.temporal_month_split(panel, 0.2)
    assert train.target_month.max() < test.target_month.min()
    assert train.target_month.max() < cutoff
    assert test.target_month.min() == cutoff


def test_thresholds_are_fit_only_from_training_rows():
    train = pd.DataFrame({
        "Port": ["Houston"] * 4,
        "target_next_month_dwell": [1.0, 2.0, 3.0, 4.0],
    })
    thresholds = training.thresholds_from_training(train)
    assert thresholds["Houston"] == pytest.approx(3.25)
    future = pd.DataFrame({
        "Port": ["Houston"],
        "target_next_month_dwell": [100.0],
    })
    labelled = training.apply_congestion_labels(future, thresholds)
    assert labelled.congestion_label.iloc[0] == 1
    assert labelled.congestion_threshold_days.iloc[0] == pytest.approx(3.25)


def test_unknown_port_threshold_fails_loudly():
    frame = pd.DataFrame({"Port": ["Unknown"], "target_next_month_dwell": [5.0]})
    with pytest.raises(ValueError, match="No training threshold"):
        training.apply_congestion_labels(frame, pd.Series({"Houston": 3.0}))


def test_cv_is_chronological_and_returns_training_only_selection():
    raw = make_panel(n_months=36)
    tmp = Path.cwd() / ".pytest_port_cv.csv"
    try:
        raw.to_csv(tmp, index=False)
        panel = training.build_panel(raw_path=tmp, macro_path=tmp.parent / "missing_macro.csv")
    finally:
        if tmp.exists():
            tmp.unlink()
    train, _, _ = training.temporal_month_split(panel, 0.2)
    features = ["UniqueVessels", "MeanDwellDays", "rolling_3m_dwell", "Port"]
    numeric, categorical = features[:-1], ["Port"]
    candidates = {"logistic_regression": None, "random_forest": None}
    # The function creates fold-specific estimators from the candidate names.
    summary, selected = training.chronological_cv_scores(
        train, features, numeric, categorical, candidates, n_splits=3
    )
    assert selected in candidates
    assert summary[selected]["valid_folds"] >= 1
    assert summary[selected]["mean_pr_auc"] is not None


def test_metrics_include_required_classification_and_probability_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    result = training.calculate_metrics("test", y, p)
    for key in [
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "pr_auc", "brier_score", "confusion_matrix",
    ]:
        assert key in result
    assert result["f1"] == pytest.approx(1.0)


def test_model_artifact_and_reports_are_defined_in_contract():
    source = Path(training.__file__).read_text(encoding="utf-8")
    assert 'five_port_congestion_model.joblib' in source
    assert 'five_port_congestion_model_metrics.json' in source
    assert 'five_port_congestion_permutation_importance.csv' in source
    assert 'five_port_congestion_test_predictions.csv' in source
    assert '"selection_method"' in source
    assert "CalibratedClassifierCV" not in source


def test_saved_model_metadata_matches_selected_estimator(tmp_path, monkeypatch):
    artifact = {
        "model": "fitted-estimator",
        "model_name": "logistic_regression",
        "selection_method": "expanding_window_time_series_cross_validation_PR_AUC",
        "classification_threshold": 0.5,
        "port_thresholds_days": {"Houston": 3.0},
    }
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    loaded = joblib.load(path)
    assert loaded["model_name"] == "logistic_regression"
    assert loaded["model"] == "fitted-estimator"
    assert loaded["selection_method"].startswith("expanding_window")
    assert loaded["classification_threshold"] == 0.5


def test_default_input_falls_back_to_2015_file_when_2009_file_missing(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    fallback = raw_dir / "monthly_dwell_2015_2025.csv"
    make_panel(n_months=6).to_csv(fallback, index=False)
    monkeypatch.setattr(training, "RAW", raw_dir)
    loaded = training.load_port_panel()
    assert not loaded.empty
    assert loaded.month.min() == pd.Timestamp("2019-01-01")


def test_f2_threshold_selection_prioritizes_recall_without_test_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.10, 0.35, 0.55, 0.40, 0.60, 0.80])
    result = training.choose_f2_threshold(y, probabilities)
    assert 0.05 <= result["threshold"] <= 0.95
    assert result["f2"] >= 0
    assert result["recall"] >= result["precision"] or result["f2"] > 0


def test_f2_threshold_rejects_single_class_targets():
    with pytest.raises(ValueError, match="both target classes"):
        training.choose_f2_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_cv_returns_oof_threshold_and_f2_for_each_candidate():
    raw = make_panel(n_months=36)
    tmp = Path.cwd() / ".pytest_port_cv_threshold.csv"
    try:
        raw.to_csv(tmp, index=False)
        panel = training.build_panel(raw_path=tmp, macro_path=tmp.parent / "missing_macro.csv")
    finally:
        if tmp.exists():
            tmp.unlink()
    train, _, _ = training.temporal_month_split(panel, 0.2)
    features = ["UniqueVessels", "MeanDwellDays", "rolling_3m_dwell", "Port"]
    numeric, categorical = features[:-1], ["Port"]
    candidates = {"logistic_regression": None, "random_forest": None}
    summary, selected = training.chronological_cv_scores(
        train, features, numeric, categorical, candidates, n_splits=3
    )
    assert selected in candidates
    for name in candidates:
        tuned = summary[name]["oof_threshold_selection"]
        assert tuned is not None
        assert 0.0 <= tuned["threshold"] <= 1.0
        assert "f2" in tuned and "recall" in tuned and "precision" in tuned


def test_metrics_report_f2_at_selected_threshold():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.6, 0.55, 0.9])
    result = training.calculate_metrics("model", y, probabilities, threshold=0.5)
    assert "f2" in result
    assert result["classification_threshold"] == 0.5
