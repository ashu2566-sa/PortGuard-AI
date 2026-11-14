
from datetime import datetime
from pathlib import Path
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api_schemas import PredictionInput, BatchPredictionInput
from src.inference import build_latest_features
from src.port_explanations import explain_prediction


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "five_port_congestion_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

ARTIFACT = joblib.load(MODEL_PATH)

MODEL = ARTIFACT["model"]
FEATURE_COLUMNS = ARTIFACT["feature_columns"]
NUMERIC_FEATURES = ARTIFACT["numeric_features"]

GLOBAL_THRESHOLD = float(ARTIFACT["classification_threshold"])
PORT_THRESHOLDS = ARTIFACT["port_classification_thresholds"]
DWELL_THRESHOLDS = ARTIFACT["port_thresholds_days"]
MODEL_NAME = ARTIFACT["model_name"]

SUPPORTED_PORTS = sorted(DWELL_THRESHOLDS.keys())

app = FastAPI(
    title="PortGuard AI API",
    description=(
        "AI-powered monthly congestion prediction and "
        "explainability service for five US ports."
    ),
    version="1.1.0",
)


def get_months(feature_month: str):
    try:
        current = datetime.strptime(feature_month, "%Y-%m")
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="feature_month must use YYYY-MM format.",
        )

    if current.strftime("%Y-%m") != feature_month:
        raise HTTPException(
            status_code=422,
            detail="feature_month must use YYYY-MM format.",
        )

    if current.month == 12:
        target = current.replace(year=current.year + 1, month=1)
    else:
        target = current.replace(month=current.month + 1)

    return current.strftime("%Y-%m"), target.strftime("%Y-%m")


def predict_one(request: PredictionInput):

    if request.port not in SUPPORTED_PORTS:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unsupported port.",
                "supported_ports": SUPPORTED_PORTS,
            },
        )

    feature_month, target_month = get_months(request.feature_month)

    if request.month_num != datetime.strptime(
        feature_month, "%Y-%m"
    ).month:
        raise HTTPException(
            status_code=422,
            detail="month_num must match the month in feature_month.",
        )

    values = {
        "UniqueVessels": request.unique_vessels,
        "MeanDwellDays": request.mean_dwell_days,
        "MedianDwellDays": request.median_dwell_days,
        "StdDwellDays": request.std_dwell_days,
        "TotalObservations": request.total_observations,
        "AvgVesselLength": request.avg_vessel_length,
        "rolling_3m_dwell": request.rolling_3m_dwell,
        "dwell_change_1m": request.dwell_change_1m,
        "vessel_change_1m": request.vessel_change_1m,
        "gscpi": request.gscpi,
        "month_num": request.month_num,
        "Port": request.port,
    }

    for feature in NUMERIC_FEATURES:
        if not math.isfinite(float(values[feature])):
            raise HTTPException(
                status_code=422,
                detail=f"{feature} must be a finite number.",
            )

    input_frame = pd.DataFrame(
        [[values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    try:
        probability = float(
            MODEL.predict_proba(input_frame)[0][1]
        )

        explanation = explain_prediction(
            MODEL,
            input_frame,
            top_n=3,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction or explanation failed: {str(exc)}",
        )

    port_config = PORT_THRESHOLDS.get(request.port, {})

    classification_threshold = float(
        port_config.get("threshold", GLOBAL_THRESHOLD)
    )

    threshold_status = port_config.get(
        "status", "global_fallback"
    )

    predicted_congestion = (
        probability >= classification_threshold
    )

    return {
        "port": request.port,
        "feature_month": feature_month,
        "predicted_target_month": target_month,
        "model_name": MODEL_NAME,
        "congestion_probability": round(probability, 6),
        "classification_threshold": classification_threshold,
        "threshold_status": threshold_status,
        "predicted_congestion": bool(predicted_congestion),
        "prediction": (
            "CONGESTION"
            if predicted_congestion
            else "NO_CONGESTION"
        ),
        "port_congestion_threshold_days": float(
            DWELL_THRESHOLDS[request.port]
        ),
        "explanation": explanation,
    }


@app.get("/")
def root():
    return {
        "service": "PortGuard AI",
        "message": "Port congestion prediction API",
        "version": "1.1.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_NAME,
        "explainability_enabled": True,
    }


@app.get("/ports")
def get_ports():
    return {
        "supported_ports": [
            {
                "port": port,
                "congestion_threshold_days": float(
                    DWELL_THRESHOLDS[port]
                ),
                "classification_threshold": float(
                    PORT_THRESHOLDS.get(
                        port,
                        {"threshold": GLOBAL_THRESHOLD},
                    )["threshold"]
                ),
            }
            for port in SUPPORTED_PORTS
        ]
    }


@app.post("/predict")
def predict(request: PredictionInput):
    return predict_one(request)


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionInput):
    results = [
        predict_one(item)
        for item in request.predictions
    ]

    return {
        "count": len(results),
        "results": results,
    }


@app.get("/predict/all-ports")
def predict_all_ports():
    try:
        latest_features = build_latest_features()

        results = []

        for _, row in latest_features.iterrows():

            request = PredictionInput(
                port=row["Port"],
                feature_month=row["month"].strftime("%Y-%m"),
                unique_vessels=float(row["UniqueVessels"]),
                mean_dwell_days=float(row["MeanDwellDays"]),
                median_dwell_days=float(row["MedianDwellDays"]),
                std_dwell_days=float(row["StdDwellDays"]),
                total_observations=float(row["TotalObservations"]),
                avg_vessel_length=float(row["AvgVesselLength"]),
                rolling_3m_dwell=float(row["rolling_3m_dwell"]),
                dwell_change_1m=float(row["dwell_change_1m"]),
                vessel_change_1m=float(row["vessel_change_1m"]),
                gscpi=float(row["gscpi"]),
                month_num=int(row["month_num"]),
            )

            results.append(predict_one(request))

        return {
            "count": len(results),
            "data_as_of_month": (
                latest_features["month"]
                .max()
                .strftime("%Y-%m")
            ),
            "forecast_month": (
                results[0]["predicted_target_month"]
            ),
            "results": results,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Automated prediction failed: {str(exc)}",
        )
