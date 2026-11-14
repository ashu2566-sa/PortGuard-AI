from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, fbeta_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

REQUIRED_COLUMNS = {
    "Port", "YearMonth", "UniqueVessels", "MeanDwellDays",
    "MedianDwellDays", "StdDwellDays", "TotalObservations",
    "AvgVesselLength",
}

# These columns are intentionally excluded: the original permutation report
# showed negative importance for both, and they encode data-coverage metadata.
EXCLUDED_FEATURES = {"days_ok", "days_total"}


def load_port_panel(path=None):
    if path is not None:
        path = Path(path)
    else:
        preferred = RAW / "monthly_dwell_2009_2025.csv"
        fallback = RAW / "monthly_dwell_2015_2025.csv"
        path = preferred if preferred.exists() else fallback
    if not path.exists():
        raise FileNotFoundError(
            "Missing monthly AIS input. Expected either "
            "data/raw/monthly_dwell_2009_2025.csv or "
            "data/raw/monthly_dwell_2015_2025.csv. Run "
            "`python -m src.download_data` only if neither file exists."
        )

    panel = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required AIS columns: {sorted(missing)}")

    panel["month"] = pd.to_datetime(
        panel["YearMonth"].astype(str) + "-01", errors="coerce"
    )
    panel = panel.dropna(subset=["month", "Port"]).copy()
    panel["Port"] = panel["Port"].astype(str)

    numeric = [
        "UniqueVessels", "MeanDwellDays", "MedianDwellDays",
        "StdDwellDays", "MinDwellDays", "MaxDwellDays",
        "TotalObservations", "AvgVesselLength",
    ]
    for col in numeric:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = (
        panel.sort_values(["Port", "month"])
        .drop_duplicates(["Port", "month"], keep="last")
        .reset_index(drop=True)
    )
    if panel.empty:
        raise ValueError("AIS input contains no valid Port/YearMonth records.")
    return panel


def build_panel(raw_path=None, macro_path=None):
    panel = load_port_panel(raw_path)
    panel = panel.sort_values(["Port", "month"]).copy()
    grouped = panel.groupby("Port", group_keys=False)

    # Features use only the current and preceding months.
    panel["rolling_3m_dwell"] = grouped["MeanDwellDays"].transform(
        lambda values: values.rolling(3, min_periods=2).mean()
    )
    panel["dwell_change_1m"] = grouped["MeanDwellDays"].diff()
    panel["vessel_change_1m"] = grouped["UniqueVessels"].pct_change()

    # Month t predicts dwell in month t+1.
    panel["target_month"] = grouped["month"].shift(-1)
    panel["target_next_month_dwell"] = grouped["MeanDwellDays"].shift(-1)

    expected = panel["month"] + pd.offsets.MonthBegin(1)
    invalid = panel["target_month"] != expected
    panel.loc[
        invalid, ["target_month", "target_next_month_dwell"]
    ] = [pd.NaT, np.nan]

    panel["month_num"] = panel["month"].dt.month

    macro_path = Path(macro_path) if macro_path else RAW / "analysis_dataset_dwell.csv"

    if macro_path.exists():
        macro = pd.read_csv(macro_path)

        if not {"date", "gscpi"}.issubset(macro.columns):
            raise ValueError("Macro file must contain 'date' and 'gscpi' columns.")

        macro["month"] = (
            pd.to_datetime(macro["date"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        macro["gscpi"] = pd.to_numeric(macro["gscpi"], errors="coerce")

        macro = (
            macro.dropna(subset=["month"])
            .sort_values("month")
            .drop_duplicates("month", keep="last")
        )

        # GSCPI for observation month t is published in month t+1.
        # At the end of forecast month t, use the latest available
        # observation: GSCPI from month t-1.
        macro["month"] = macro["month"] + pd.offsets.MonthBegin(1)

        macro = macro[["month", "gscpi"]]

        panel = panel.merge(
            macro,
            on="month",
            how="left",
            validate="many_to_one"
        )
    else:
        panel["gscpi"] = np.nan

    panel = panel.dropna(
        subset=["target_month", "target_next_month_dwell"]
    ).copy()

    return panel.reset_index(drop=True)



def temporal_month_split(panel, test_fraction=0.20):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    months = np.array(sorted(panel["target_month"].dropna().unique()))
    if len(months) < 5:
        raise ValueError("At least five distinct target months are required.")
    cut = int(np.floor(len(months) * (1 - test_fraction)))
    cut = min(max(cut, 1), len(months) - 1)
    cutoff = pd.Timestamp(months[cut])
    train = panel.loc[panel.target_month < cutoff].copy()
    test = panel.loc[panel.target_month >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    if train.target_month.max() >= test.target_month.min():
        raise AssertionError("Temporal split overlap detected.")
    return train, test, cutoff


def thresholds_from_training(frame, quantile=0.75):
    if frame.empty:
        raise ValueError("Cannot calculate congestion thresholds from empty data.")
    return frame.groupby("Port")["target_next_month_dwell"].quantile(quantile)


def apply_congestion_labels(frame, thresholds):
    result = frame.copy()
    result["congestion_threshold_days"] = result["Port"].map(thresholds)
    if result["congestion_threshold_days"].isna().any():
        unknown = sorted(result.loc[
            result.congestion_threshold_days.isna(), "Port"
        ].astype(str).unique())
        raise ValueError(f"No training threshold available for ports: {unknown}")
    result["congestion_label"] = (
        result["target_next_month_dwell"] > result["congestion_threshold_days"]
    ).astype(int)
    return result


def make_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def make_candidates(numeric, categorical, positive_weight):
    return {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=42
            )),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", XGBClassifier(
                n_estimators=350, max_depth=3, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=2,
                scale_pos_weight=positive_weight, eval_metric="logloss",
                random_state=42, n_jobs=-1
            )),
        ]),
    }


def make_time_cv(train, n_splits=3):
    months = np.array(sorted(train.target_month.unique()))
    if len(months) <= n_splits:
        return []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_month_idx, valid_month_idx in splitter.split(months):
        train_months = set(months[train_month_idx])
        valid_months = set(months[valid_month_idx])
        train_idx = np.flatnonzero(train.target_month.isin(train_months).to_numpy())
        valid_idx = np.flatnonzero(train.target_month.isin(valid_months).to_numpy())
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds


def choose_f2_threshold(y_true, probabilities):
    """Choose a recall-focused threshold from training-only OOF probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("Threshold selection requires matching, non-empty arrays.")
    if np.unique(y_true).size < 2:
        raise ValueError("Threshold selection requires both target classes.")

    candidates = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 91), probabilities
    ]))
    best = None
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        f2 = fbeta_score(y_true, predicted, beta=2, zero_division=0)
        precision = precision_score(y_true, predicted, zero_division=0)
        recall = recall_score(y_true, predicted, zero_division=0)
        candidate = (float(f2), float(precision), float(threshold), float(recall))
        # Prefer higher recall-focused F2; on ties prefer precision, then higher threshold.
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return {
        "threshold": best[2],
        "f2": best[0],
        "precision": best[1],
        "recall": best[3],
    }


def choose_port_thresholds(y_true, probabilities, ports, fallback_threshold,
                           min_precision=0.60, min_positive_support=8):
    """Tune each port threshold from training-only OOF predictions.

    If a port has too few OOF positives or no threshold meets the precision
    constraint, use the global OOF threshold and explicitly report the fallback.
    """
    frame = pd.DataFrame({
        "y": np.asarray(y_true, dtype=int),
        "probability": np.asarray(probabilities, dtype=float),
        "Port": np.asarray(ports, dtype=str),
    })
    if frame.empty or frame.isna().any().any():
        raise ValueError("Port threshold selection requires complete OOF data.")

    results = {}
    for port, group in frame.groupby("Port"):
        positives = int(group.y.sum())
        if positives < min_positive_support or group.y.nunique() < 2:
            results[port] = {
                "threshold": float(fallback_threshold),
                "status": "global_fallback_insufficient_support",
                "oof_positives": positives,
                "precision_constraint": float(min_precision),
            }
            continue

        candidates = np.unique(np.concatenate([
            np.linspace(0.05, 0.95, 91), group.probability.to_numpy()
        ]))
        feasible = []
        for threshold in candidates:
            pred = (group.probability.to_numpy() >= threshold).astype(int)
            precision = precision_score(group.y, pred, zero_division=0)
            recall = recall_score(group.y, pred, zero_division=0)
            f2 = fbeta_score(group.y, pred, beta=2, zero_division=0)
            if precision >= min_precision:
                feasible.append((float(f2), float(recall), float(precision), float(threshold)))

        if not feasible:
            results[port] = {
                "threshold": float(fallback_threshold),
                "status": "global_fallback_precision_constraint_unattainable",
                "oof_positives": positives,
                "precision_constraint": float(min_precision),
            }
        else:
            best = max(feasible)
            results[port] = {
                "threshold": best[3],
                "status": "port_specific_threshold_selected",
                "oof_positives": positives,
                "oof_f2": best[0],
                "oof_recall": best[1],
                "oof_precision": best[2],
                "precision_constraint": float(min_precision),
            }
    return results


def calculate_port_metrics(frame):
    """Return test-only diagnostics; these metrics never tune thresholds."""
    results = {}
    for port, group in frame.groupby("Port"):
        y = group.congestion_label.astype(int)
        pred = group.predicted_congestion.astype(int)
        results[str(port)] = {
            "records": int(len(group)),
            "actual_congestion": int(y.sum()),
            "predicted_congestion": int(pred.sum()),
            "true_positive": int(((y == 1) & (pred == 1)).sum()),
            "false_positive": int(((y == 0) & (pred == 1)).sum()),
            "false_negative": int(((y == 1) & (pred == 0)).sum()),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        }
    return results


def chronological_cv_scores(train, features, numeric, categorical, candidates, n_splits=3):
    """Compare models and tune thresholds using expanding-window CV within outer training."""
    folds = make_time_cv(train, n_splits=n_splits)
    if len(folds) < 2:
        raise ValueError(
            "Insufficient distinct training months for at least two temporal CV folds."
        )

    fold_pr_auc = {name: [] for name in candidates}
    oof_y = {name: [] for name in candidates}
    oof_probabilities = {name: [] for name in candidates}
    oof_ports = {name: [] for name in candidates}
    for train_idx, valid_idx in folds:
        fold_train = train.iloc[train_idx].copy()
        fold_valid = train.iloc[valid_idx].copy()

        fold_thresholds = thresholds_from_training(fold_train)
        fold_train = apply_congestion_labels(fold_train, fold_thresholds)
        fold_valid = apply_congestion_labels(fold_valid, fold_thresholds)
        y_fold_train = fold_train.congestion_label
        y_fold_valid = fold_valid.congestion_label

        if y_fold_train.nunique() < 2 or y_fold_valid.nunique() < 2:
            continue

        pos_weight = (
            (y_fold_train == 0).sum() / max(1, (y_fold_train == 1).sum())
        )
        fold_candidates = make_candidates(numeric, categorical, pos_weight)
        for name in candidates:
            estimator = fold_candidates[name]
            estimator.fit(fold_train[features], y_fold_train)
            probabilities = estimator.predict_proba(fold_valid[features])[:, 1]
            fold_pr_auc[name].append(
                average_precision_score(y_fold_valid, probabilities)
            )
            oof_y[name].extend(y_fold_valid.astype(int).tolist())
            oof_probabilities[name].extend(probabilities.astype(float).tolist())
            oof_ports[name].extend(fold_valid.Port.astype(str).tolist())

    summary = {}
    for name in candidates:
        if oof_y[name] and len(np.unique(oof_y[name])) == 2:
            threshold_result = choose_f2_threshold(oof_y[name], oof_probabilities[name])
            port_thresholds = choose_port_thresholds(
                oof_y[name], oof_probabilities[name], oof_ports[name],
                fallback_threshold=threshold_result["threshold"],
                min_precision=0.60,
                min_positive_support=8,
            )
            mean_pr_auc = float(np.mean(fold_pr_auc[name])) if fold_pr_auc[name] else None
        else:
            port_thresholds = None
            threshold_result = None
            mean_pr_auc = None
        summary[name] = {
            "fold_pr_auc": fold_pr_auc[name],
            "mean_pr_auc": mean_pr_auc,
            "valid_folds": len(fold_pr_auc[name]),
            "oof_threshold_selection": threshold_result,
            "oof_port_thresholds": port_thresholds,
        }

    eligible = {
        name: details["oof_threshold_selection"]["f2"]
        for name, details in summary.items()
        if details["oof_threshold_selection"] is not None
    }
    if not eligible:
        raise ValueError(
            "No valid CV folds contained both classes. More history is required."
        )
    best_f2 = max(eligible.values())
    tied = [name for name, value in eligible.items() if np.isclose(value, best_f2)]
    # PR-AUC is the tie-breaker; test-set results are never consulted.
    best_name = max(
        tied,
        key=lambda name: summary[name]["mean_pr_auc"] or -1.0,
    )
    return summary, best_name


def calculate_metrics(name, y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": name,
        "classification_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if pd.Series(y_true).nunique() > 1 else None
        ),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def main():
    panel = build_panel()
    train_raw, test_raw, cutoff = temporal_month_split(panel)

    # Outer-test labels are derived only from thresholds fitted on outer training.
    thresholds = thresholds_from_training(train_raw)
    train = apply_congestion_labels(train_raw, thresholds)
    test = apply_congestion_labels(test_raw, thresholds)

    numeric = [
        col for col in [
            "UniqueVessels", "MeanDwellDays", "MedianDwellDays",
            "StdDwellDays", "TotalObservations", "AvgVesselLength",
            "rolling_3m_dwell", "dwell_change_1m", "vessel_change_1m",
            "gscpi", "month_num",
        ] if col in panel.columns and col not in EXCLUDED_FEATURES
    ]
    categorical = ["Port"]
    features = numeric + categorical

    X_train, y_train = train[features], train.congestion_label
    X_test, y_test = test[features], test.congestion_label
    if y_train.nunique() < 2:
        raise ValueError("Training period must contain both congestion classes.")
    if y_test.nunique() < 2:
        warnings.warn("Test period contains only one class; ROC-AUC will be unavailable.")

    positive_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    candidates = make_candidates(numeric, categorical, positive_weight)

    # Model selection and threshold tuning use only chronological OOF predictions
    # generated inside the outer training period. The outer test remains untouched.
    cv_summary, best_name = chronological_cv_scores(
        train, features, numeric, categorical, candidates
    )
    selected_threshold = cv_summary[best_name]["oof_threshold_selection"]["threshold"]
    selected_port_thresholds = cv_summary[best_name]["oof_port_thresholds"]

    # Refit every candidate on the full outer training period for transparent
    # final-test comparison. The selected artifact is chosen by CV, not test metrics.
    test_metrics = []
    fitted = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        fitted[name] = estimator
        probabilities = estimator.predict_proba(X_test)[:, 1]
        threshold = cv_summary[name]["oof_threshold_selection"]["threshold"]
        test_metrics.append(calculate_metrics(name, y_test, probabilities, threshold))

    best = fitted[best_name]
    best_probabilities = best.predict_proba(X_test)[:, 1]
    selected_test = test.copy()
    selected_test["congestion_probability"] = best_probabilities
    selected_test["classification_threshold"] = selected_test["Port"].map(
        {port: details["threshold"] for port, details in selected_port_thresholds.items()}
    ).fillna(selected_threshold)
    selected_test["predicted_congestion"] = (
        selected_test["congestion_probability"] >= selected_test["classification_threshold"]
    ).astype(int)
    selected_metrics = calculate_metrics(
        best_name, y_test, best_probabilities, selected_threshold
    )
    selected_predictions = selected_test["predicted_congestion"].to_numpy()
    selected_metrics.update({
        "threshold_strategy": "port_specific_with_global_fallback",
        "classification_threshold": "port_specific; see port_classification_thresholds",
        "accuracy": float(accuracy_score(y_test, selected_predictions)),
        "precision": float(precision_score(y_test, selected_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, selected_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, selected_predictions, zero_division=0)),
        "f2": float(fbeta_score(y_test, selected_predictions, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, selected_predictions).tolist(),
        "port_level_metrics": calculate_port_metrics(selected_test),
    })

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": best,
        "feature_columns": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "port_thresholds_days": thresholds.to_dict(),
        "target_definition": (
            "next_month_mean_dwell_above_port_specific_training_75th_percentile"
        ),
        "model_name": best_name,
        "classification_threshold": selected_threshold,
        "port_classification_thresholds": selected_port_thresholds,
        "selection_method": "expanding_window_time_series_cross_validation_OOF_F2_then_PR_AUC_with_port_thresholds",
        "cutoff_target_month": str(cutoff.date()),
    }
    joblib.dump(model_artifact, MODELS / "five_port_congestion_model.joblib")

    importance = permutation_importance(
        best, X_test, y_test, n_repeats=10,
        random_state=42, scoring="average_precision", n_jobs=-1
    )
    importance_frame = pd.DataFrame({
        "feature": features,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(
        REPORTS / "five_port_congestion_permutation_importance.csv", index=False
    )

    predictions = test[[
        "Port", "month", "target_month", "target_next_month_dwell",
        "congestion_threshold_days", "congestion_label",
    ]].copy()
    predictions["congestion_probability"] = best_probabilities
    predictions["classification_threshold"] = selected_test["classification_threshold"].to_numpy()
    predictions["predicted_congestion"] = selected_test["predicted_congestion"].to_numpy()
    predictions.to_csv(
        PROCESSED / "five_port_congestion_test_predictions.csv", index=False
    )

    summary = {
        "records": len(panel),
        "ports": sorted(panel.Port.unique().tolist()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_target_month_end": str(train.target_month.max().date()),
        "test_target_month_start": str(test.target_month.min().date()),
        "cutoff_target_month": str(cutoff.date()),
        "port_thresholds_days": thresholds.to_dict(),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "selected_model": best_name,
        "selection_method": "expanding_window_time_series_cross_validation_OOF_F2_then_PR_AUC",
        "selected_classification_threshold": selected_threshold,
        "selected_port_classification_thresholds": selected_port_thresholds,
        "port_level_test_metrics": calculate_port_metrics(selected_test),
        "cross_validation": cv_summary,
        "selected_model_test_metrics": selected_metrics,
        "test_model_comparison": test_metrics,
        "top_permutation_features": importance_frame.head(10).to_dict(orient="records"),
        "notes": [
            "The final test set was not used for model or threshold selection.",
            "Model selection prioritizes pooled chronological out-of-fold F2; PR-AUC breaks ties.",
            "Global and port-specific classification thresholds are tuned on training-only out-of-fold probabilities.",
            "Port thresholds require at least 8 OOF positives and 60% OOF precision; otherwise the global threshold is used and the fallback is reported.",
            "Congestion thresholds were fitted on training data only.",
            "CV fold labels use thresholds fitted separately within each fold.",
            "Calibration was not applied because the prior calibrated model "
            "materially degraded classification performance.",
            "GSCPI observation-month values are shifted forward one calendar month so a forecast row for month t uses the prior month t-1 observation.",
            "Historical GSCPI revisions/vintages are not represented in the source CSV; point-in-time vintage validation is still required before operational use.",
        ],
    }
    (REPORTS / "five_port_congestion_model_metrics.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
