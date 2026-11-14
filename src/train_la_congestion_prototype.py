from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"
MODELS=ROOT/"models"
REPORTS=ROOT/"reports"
PROCESSED=ROOT/"data"/"processed"

def main():
    path=RAW/"analysis_dataset_dwell.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing uploaded data: {path}")
    df=pd.read_csv(path,parse_dates=["date"])
    df=df.dropna(subset=["dwell_la_raw"]).sort_values("date").copy()
    df["target_month"]=df["date"]+pd.offsets.MonthBegin(1)
    df["target_dwell_days"]=df["dwell_la_raw"].shift(-1)
    df["rolling_3m_dwell"]=df["dwell_la_raw"].rolling(3,min_periods=2).mean()
    df["dwell_change_1m"]=df["dwell_la_raw"].diff()
    df["vessel_change_1m"]=df["la_unique_vessels"].diff()
    df["month_num"]=df["date"].dt.month
    df=df.dropna(subset=["target_dwell_days"]).reset_index(drop=True)

    # The threshold is fitted only on the training portion.
    split=int(np.floor(len(df)*0.8))
    train,test=df.iloc[:split].copy(),df.iloc[split:].copy()
    threshold=float(train["target_dwell_days"].quantile(0.75))
    train["congestion_label"]=(train.target_dwell_days>threshold).astype(int)
    test["congestion_label"]=(test.target_dwell_days>threshold).astype(int)

    features=[
        "dwell_la_raw","la_unique_vessels","dwell_la_detrended",
        "dwell_composite","rolling_3m_dwell","dwell_change_1m",
        "vessel_change_1m","gscpi","cpi_yoy","cpi_goods_yoy",
        "cpi_services_yoy","indpro_yoy","oil_yoy","month_num"
    ]
    pre=ColumnTransformer([("numeric",Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ]),features)])
    candidates={
        "logistic_regression":Pipeline([
            ("prep",pre),
            ("model",LogisticRegression(max_iter=2000,class_weight="balanced"))
        ]),
        "random_forest":Pipeline([
            ("prep",pre),
            ("model",RandomForestClassifier(
                n_estimators=300,min_samples_leaf=3,
                class_weight="balanced",random_state=42,n_jobs=-1
            ))
        ])
    }
    rows=[]; fitted={}
    for name,model in candidates.items():
        model.fit(train[features],train.congestion_label)
        probability=model.predict_proba(test[features])[:,1]
        predicted=(probability>=0.5).astype(int)
        rows.append({
            "model":name,
            "accuracy":accuracy_score(test.congestion_label,predicted),
            "precision":precision_score(test.congestion_label,predicted,zero_division=0),
            "recall":recall_score(test.congestion_label,predicted,zero_division=0),
            "f1":f1_score(test.congestion_label,predicted,zero_division=0),
            "roc_auc":roc_auc_score(test.congestion_label,probability),
            "pr_auc":average_precision_score(test.congestion_label,probability),
            "brier_score":brier_score_loss(test.congestion_label,probability),
            "confusion_matrix":confusion_matrix(test.congestion_label,predicted).tolist()
        })
        fitted[name]=model

    comparison=pd.DataFrame(rows).sort_values(
        ["pr_auc","roc_auc","f1"],ascending=False
    )
    best_name=comparison.iloc[0].model
    selected=fitted[best_name]
    probability=selected.predict_proba(test[features])[:,1]

    MODELS.mkdir(parents=True,exist_ok=True)
    REPORTS.mkdir(parents=True,exist_ok=True)
    PROCESSED.mkdir(parents=True,exist_ok=True)
    joblib.dump({
        "model":selected,"feature_columns":features,
        "model_name":best_name,"threshold_dwell_days":threshold,
        "probability_calibrated":False,
        "target_definition":"next_month_mean_dwell_above_training_75th_percentile"
    },MODELS/"la_port_congestion_model.joblib")

    permutation=permutation_importance(
        selected,test[features],test.congestion_label,
        n_repeats=10,random_state=42,scoring="average_precision",n_jobs=-1
    )
    importance=pd.DataFrame({
        "feature":features,
        "importance_mean":permutation.importances_mean,
        "importance_std":permutation.importances_std
    }).sort_values("importance_mean",ascending=False)
    importance.to_csv(REPORTS/"la_port_congestion_permutation_importance.csv",index=False)

    predictions=test[["date","target_month","target_dwell_days","congestion_label"]].copy()
    predictions["congestion_probability_uncalibrated"]=probability
    predictions["predicted_congestion"]=(probability>=0.5).astype(int)
    predictions.to_csv(PROCESSED/"la_port_congestion_test_predictions.csv",index=False)

    summary={
        "port":"Los Angeles / Long Beach",
        "records":len(df),"train_records":len(train),"test_records":len(test),
        "train_target_month_start":str(train.target_month.min().date()),
        "train_target_month_end":str(train.target_month.max().date()),
        "test_target_month_start":str(test.target_month.min().date()),
        "test_target_month_end":str(test.target_month.max().date()),
        "congestion_threshold_dwell_days":threshold,
        "train_positive_rate":float(train.congestion_label.mean()),
        "test_positive_rate":float(test.congestion_label.mean()),
        "selected_model":best_name,
        "probability_calibrated":False,
        "metrics":comparison.to_dict(orient="records"),
        "top_permutation_features":importance.head(10).to_dict(orient="records"),
        "limitation":"Single-port prototype. Validate with five-port AIS panel before generalizing."
    }
    (REPORTS/"la_port_congestion_model_metrics.json").write_text(
        json.dumps(summary,indent=2,default=float),encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,default=float))

if __name__=="__main__":
    main()
