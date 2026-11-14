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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TARGET = "Disruption_Occurred"
NUMERIC = [
    "Distance_km", "Weight_MT", "Fuel_Price_Index",
    "Geopolitical_Risk_Score", "Carrier_Reliability_Score",
    "month_num", "day_of_week", "day_of_year",
]
CATEGORICAL = [
    "Origin_Port", "Destination_Port", "Transport_Mode",
    "Product_Category", "Weather_Condition",
]
HYBRID_NUMERIC = [
    "la_mean_dwell_lag1", "la_dwell_composite_lag1",
    "la_unique_vessels_lag1", "la_dwell_detrended_lag1",
    "gscpi_lag1", "cpi_yoy_lag1", "cpi_goods_yoy_lag1",
    "cpi_services_yoy_lag1", "indpro_yoy_lag1", "oil_yoy_lag1",
]

def make_preprocessor(numeric, categorical, scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline(categorical_steps), categorical),
    ])

def temporal_split_by_date(frame, date_col="Date", test_fraction=0.20):
    """Keep every record from the same calendar date in one partition."""
    unique_dates = np.array(sorted(frame[date_col].dropna().unique()))
    cutoff_index = max(1, int(np.floor(len(unique_dates) * (1-test_fraction))))
    cutoff = pd.Timestamp(unique_dates[cutoff_index])
    train = frame.loc[frame[date_col] < cutoff].copy()
    test = frame.loc[frame[date_col] >= cutoff].copy()
    if set(train[date_col]).intersection(set(test[date_col])):
        raise AssertionError("A calendar date appears in both train and test.")
    return train, test, cutoff


def make_date_group_cv(train, n_splits=3):
    """Create expanding folds without splitting records from the same date."""
    dates = np.array(sorted(train["Date"].unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_date_idx, valid_date_idx in splitter.split(dates):
        train_dates = set(dates[train_date_idx])
        valid_dates = set(dates[valid_date_idx])
        train_idx = np.flatnonzero(train["Date"].isin(train_dates).to_numpy())
        valid_idx = np.flatnonzero(train["Date"].isin(valid_dates).to_numpy())
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds

def metric_row(name, y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability) if y_true.nunique() > 1 else None,
        "pr_auc": average_precision_score(y_true, probability),
        "brier_score": brier_score_loss(y_true, probability),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
    }

def prepare_frame(df):
    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["month_num"] = frame["Date"].dt.month
    frame["day_of_week"] = frame["Date"].dt.dayofweek
    frame["day_of_year"] = frame["Date"].dt.dayofyear
    return frame.dropna(subset=["Date", TARGET]).sort_values("Date").reset_index(drop=True)

def build_hybrid_data(shipments, macro):
    data = shipments.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["month"] = data["Date"].dt.to_period("M").dt.to_timestamp()
    data = data[
        data["Origin_Port"].eq("Los Angeles")
        | data["Destination_Port"].eq("Los Angeles")
    ].copy()

    panel = macro.copy()
    panel["month"] = (
        pd.to_datetime(panel["date"], errors="coerce")
        .dt.to_period("M").dt.to_timestamp()
    )
    # A shipment in month M gets the last complete panel observation from M-1.
    panel["month"] = panel["month"] + pd.offsets.MonthBegin(1)
    panel = panel.rename(columns={
        "dwell_la_raw": "la_mean_dwell_lag1",
        "dwell_composite": "la_dwell_composite_lag1",
        "la_unique_vessels": "la_unique_vessels_lag1",
        "dwell_la_detrended": "la_dwell_detrended_lag1",
        "gscpi": "gscpi_lag1",
        "cpi_yoy": "cpi_yoy_lag1",
        "cpi_goods_yoy": "cpi_goods_yoy_lag1",
        "cpi_services_yoy": "cpi_services_yoy_lag1",
        "indpro_yoy": "indpro_yoy_lag1",
        "oil_yoy": "oil_yoy_lag1",
    })
    columns = ["month"] + HYBRID_NUMERIC
    panel = panel[columns].drop_duplicates("month")
    return data.merge(panel, on="month", how="left")

def run_experiment(df, name, hybrid=False):
    frame = prepare_frame(df)
    numeric = NUMERIC + (HYBRID_NUMERIC if hybrid else [])
    categorical = CATEGORICAL
    features = numeric + categorical

    # Hybrid training/evaluation requires an observed prior-month dwell value.
    if hybrid:
        frame = frame.dropna(subset=["la_mean_dwell_lag1"]).copy()

    train, test, cutoff = temporal_split_by_date(frame)
    X_train, y_train = train[features], train[TARGET].astype(int)
    X_test, y_test = test[features], test[TARGET].astype(int)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(f"{name}: both target classes are required in train and test.")

    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    candidates = {
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(numeric, categorical, scale=True)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42, n_jobs=-1
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06,
                l2_regularization=1.0, random_state=42
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", make_preprocessor(numeric, categorical)),
            ("model", XGBClassifier(
                n_estimators=350, max_depth=3, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=2,
                scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                random_state=42, n_jobs=-1
            )),
        ]),
    }

    rows, fitted = [], {}
    for model_name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        probability = estimator.predict_proba(X_test)[:, 1]
        rows.append(metric_row(model_name, y_test, probability))
        fitted[model_name] = estimator

    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    rows.append(metric_row(
        "majority_prior_baseline", y_test,
        baseline.predict_proba(X_test)[:, 1]
    ))
    comparison = pd.DataFrame(rows).sort_values(
        ["pr_auc", "roc_auc", "f1"], ascending=False
    )
    best_name = comparison.iloc[0]["model"]
    if best_name == "majority_prior_baseline":
        best_name = comparison.iloc[1]["model"]

    # Calibrate using expanding chronological folds within the training period.
    calibration_folds = make_date_group_cv(train, n_splits=3)
    calibrated = CalibratedClassifierCV(
        estimator=fitted[best_name],
        method="sigmoid",
        cv=calibration_folds,
    )
    try:
        calibrated.fit(X_train, y_train)
        selected = calibrated
        probability = selected.predict_proba(X_test)[:, 1]
        comparison = pd.concat([
            comparison,
            pd.DataFrame([metric_row(
                best_name + "_calibrated", y_test, probability
            )])
        ], ignore_index=True)
    except Exception as exc:
        warnings.warn(f"Calibration skipped for {name}: {exc}")
        selected = fitted[best_name]
        probability = selected.predict_proba(X_test)[:, 1]

    MODELS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": selected,
        "feature_columns": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": TARGET,
        "model_name": best_name,
        "hybrid": hybrid,
        "decision_threshold": 0.5,
    }, MODELS / f"{name}_model.joblib")

    importance = permutation_importance(
        selected, X_test, y_test, n_repeats=10,
        random_state=42, scoring="average_precision", n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(REPORTS / f"{name}_permutation_importance.csv", index=False)

    predictions = test[["Shipment_ID", "Date", TARGET]].copy()
    predictions["predicted_disruption_probability"] = probability
    predictions["predicted_disruption"] = (probability >= 0.5).astype(int)
    predictions.to_csv(PROCESSED / f"{name}_test_predictions.csv", index=False)

    summary = {
        "experiment": name,
        "records": len(frame),
        "train_records": len(train),
        "test_records": len(test),
        "train_start": str(train.Date.min().date()),
        "train_end": str(train.Date.max().date()),
        "test_start": str(test.Date.min().date()),
        "test_end": str(test.Date.max().date()),
        "cutoff_date": str(cutoff.date()),
        "same_date_cross_split": bool(set(train.Date).intersection(set(test.Date))),
        "train_target_rate": float(y_train.mean()),
        "test_target_rate": float(y_test.mean()),
        "selected_base_model": best_name,
        "results": comparison.to_dict(orient="records"),
        "top_permutation_features": importance_df.head(12).to_dict(orient="records"),
    }
    (REPORTS / f"{name}_model_metrics.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    return summary

def main():
    shipments = pd.read_csv(RAW / "global_supply_chain_risk_2026.csv")
    base_summary = run_experiment(shipments, "shipment_disruption", hybrid=False)
    print(json.dumps(base_summary, indent=2, default=float))

    macro_path = RAW / "analysis_dataset_dwell.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path)
        hybrid = build_hybrid_data(shipments, macro)
        hybrid_summary = run_experiment(
            hybrid, "hybrid_la_shipment_disruption", hybrid=True
        )
        print(json.dumps(hybrid_summary, indent=2, default=float))
    else:
        print("No macro/AIS panel found; hybrid experiment skipped.")

if __name__ == "__main__":
    main()
