from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"
MODELS=ROOT/"models"
REPORTS=ROOT/"reports"

def shipment_features(df):
    data=df.copy()
    data["Date"]=pd.to_datetime(data["Date"],errors="coerce")
    data["month_num"]=data.Date.dt.month
    data["day_of_week"]=data.Date.dt.dayofweek
    data["day_of_year"]=data.Date.dt.dayofyear
    return data

def hybrid_features(shipments, macro, numeric_features):
    data=shipments.copy()
    data["Date"]=pd.to_datetime(data["Date"],errors="coerce")
    data["month"]=data.Date.dt.to_period("M").dt.to_timestamp()
    data=data[
        data.Origin_Port.eq("Los Angeles")
        | data.Destination_Port.eq("Los Angeles")
    ].copy()
    panel=macro.copy()
    panel["month"]=pd.to_datetime(panel.date,errors="coerce").dt.to_period("M").dt.to_timestamp()
    panel["month"]=panel["month"]+pd.offsets.MonthBegin(1)
    panel=panel.rename(columns={
        "dwell_la_raw":"la_mean_dwell_lag1",
        "dwell_composite":"la_dwell_composite_lag1",
        "la_unique_vessels":"la_unique_vessels_lag1",
        "dwell_la_detrended":"la_dwell_detrended_lag1",
        "gscpi":"gscpi_lag1","cpi_yoy":"cpi_yoy_lag1",
        "cpi_goods_yoy":"cpi_goods_yoy_lag1",
        "cpi_services_yoy":"cpi_services_yoy_lag1",
        "indpro_yoy":"indpro_yoy_lag1","oil_yoy":"oil_yoy_lag1",
    })
    data=data.merge(panel[["month"]+numeric_features[-10:]],on="month",how="left")
    data=data.dropna(subset=["la_mean_dwell_lag1"])
    data["month_num"]=data.Date.dt.month
    data["day_of_week"]=data.Date.dt.dayofweek
    data["day_of_year"]=data.Date.dt.dayofyear
    return data

def explain_model(experiment_name, hybrid=False):
    bundle=joblib.load(MODELS/f"{experiment_name}_model.joblib")
    model=bundle["model"]
    features=bundle["feature_columns"]
    if hybrid:
        shipments=pd.read_csv(RAW/"global_supply_chain_risk_2026.csv")
        macro=pd.read_csv(RAW/"analysis_dataset_dwell.csv")
        data=hybrid_features(shipments,macro,bundle["numeric_features"])
        split_date=pd.Timestamp("2025-06-16")
    else:
        shipments=pd.read_csv(RAW/"global_supply_chain_risk_2026.csv")
        data=shipment_features(shipments)
        split_date=pd.Timestamp("2025-08-07")

    train=data[data.Date<split_date]
    test=data[data.Date>=split_date]
    background=train[features].sample(min(40,len(train)),random_state=42)
    example=test[features].iloc[[0]]

    # Explain the first fitted base estimator inside the calibrated ensemble.
    if hasattr(model,"calibrated_classifiers_"):
        base=model.calibrated_classifiers_[0].estimator
    else:
        base=model
    preprocessor=base.named_steps["prep"]
    classifier=base.named_steps["model"]
    transformed_background=np.asarray(preprocessor.transform(background))
    transformed_example=np.asarray(preprocessor.transform(example))
    transformed_names=preprocessor.get_feature_names_out()

    if classifier.__class__.__name__ == "LogisticRegression":
        explainer=shap.LinearExplainer(classifier,transformed_background)
        shap_values=np.asarray(explainer.shap_values(transformed_example)).reshape(-1)
        output_space="log-odds contribution from first calibrated fold's base estimator"
    else:
        explainer=shap.TreeExplainer(
            classifier,
            data=transformed_background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        values=np.asarray(explainer(transformed_example).values)
        shap_values=values[0,:,1] if values.ndim==3 else values.reshape(-1)
        output_space="probability contribution from first calibrated fold's base estimator"

    local=pd.DataFrame({
        "transformed_feature":transformed_names,
        "shap_value":shap_values,
        "absolute_shap_value":np.abs(shap_values),
    }).sort_values("absolute_shap_value",ascending=False)
    REPORTS.mkdir(parents=True,exist_ok=True)
    local.to_csv(REPORTS/f"{experiment_name}_local_shap.csv",index=False)

    record=test.iloc[0]
    payload={
        "experiment":experiment_name,
        "shipment_id":str(record.get("Shipment_ID","")),
        "date":str(pd.Timestamp(record["Date"]).date()),
        "actual_disruption":int(record["Disruption_Occurred"]),
        "calibrated_model_probability":float(model.predict_proba(example)[:,1][0]),
        "base_estimator_probability":float(base.predict_proba(example)[:,1][0]),
        "shap_output_space":output_space,
        "top_factors":local.head(10).to_dict(orient="records"),
        "interpretation_note":"SHAP values explain the first base estimator in the calibrated ensemble; the displayed calibrated probability is produced by the full ensemble. Treat this as a local directional explanation, not a causal explanation."
    }
    (REPORTS/f"{experiment_name}_local_shap.json").write_text(
        json.dumps(payload,indent=2,default=float),encoding="utf-8"
    )
    print(json.dumps(payload,indent=2,default=float))

def main():
    explain_model("shipment_disruption",hybrid=False)
    explain_model("hybrid_la_shipment_disruption",hybrid=True)

if __name__=="__main__":
    main()
