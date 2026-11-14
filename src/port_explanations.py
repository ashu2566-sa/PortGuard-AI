
import numpy as np


def explain_prediction(model, input_frame, top_n=3):
    """Explain a Logistic Regression pipeline prediction using exact log-odds contributions."""

    preprocessor = model.named_steps["prep"]
    classifier = model.named_steps["model"]

    transformed = preprocessor.transform(input_frame)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    contributions = transformed[0] * coefficients

    explanation = [
        {
            "feature": name.replace("numeric__", "").replace(
                "categorical__", ""
            ),
            "contribution": round(float(value), 6),
            "direction": (
                "increases_congestion_score"
                if value > 0
                else "decreases_congestion_score"
                if value < 0
                else "neutral"
            ),
        }
        for name, value in zip(feature_names, contributions)
    ]

    positive = sorted(
        [item for item in explanation if item["contribution"] > 0],
        key=lambda item: item["contribution"],
        reverse=True,
    )[:top_n]

    negative = sorted(
        [item for item in explanation if item["contribution"] < 0],
        key=lambda item: item["contribution"],
    )[:top_n]

    decision_score = float(classifier.decision_function(transformed)[0])

    return {
        "explanation_method": "logistic_regression_coefficient_contributions",
        "output_space": "log_odds",
        "decision_score": round(decision_score, 6),
        "intercept": round(float(classifier.intercept_[0]), 6),
        "top_risk_increasing_factors": positive,
        "top_risk_decreasing_factors": negative,
        "interpretation": (
            "Positive contributions increase the model's congestion score; "
            "negative contributions decrease it. These are model associations, "
            "not causal effects."
        ),
    }
