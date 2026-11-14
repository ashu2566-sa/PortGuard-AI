
import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="PortGuard AI",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 PortGuard AI")
st.subheader("Monthly Port Congestion Prediction Dashboard")
st.caption(
    "AI-powered congestion predictions and model explanations "
    "across five US ports"
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", API_URL).rstrip("/")
    refresh = st.button("Refresh predictions", type="primary")

@st.cache_data(ttl=60)
def fetch_predictions(url):
    response = requests.get(
        f"{url}/predict/all-ports",
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def format_port(port):
    return port.replace("_", " / ")


def format_feature(feature):
    names = {
        "AvgVesselLength": "Average vessel length",
        "MeanDwellDays": "Mean dwell time",
        "MedianDwellDays": "Median dwell time",
        "StdDwellDays": "Dwell time variability",
        "UniqueVessels": "Unique vessels",
        "TotalObservations": "Total observations",
        "rolling_3m_dwell": "3-month rolling dwell time",
        "dwell_change_1m": "Monthly dwell time change",
        "vessel_change_1m": "Monthly vessel count change",
        "gscpi": "Global supply chain pressure",
        "month_num": "Calendar month",
    }
    return names.get(feature, feature.replace("Port_", "Port: "))


def show_factors(factors, positive):
    if not factors:
        st.info("No factors available.")
        return

    for factor in factors:
        name = format_feature(factor["feature"])
        contribution = factor["contribution"]

        st.write(
            f"**{name}** — {contribution:+.4f}"
        )


if refresh:
    fetch_predictions.clear()

try:
    data = fetch_predictions(api_url)
    results = data["results"]

    if not results:
        st.warning("No predictions returned.")
        st.stop()

    df = pd.DataFrame(results)

    st.info(
        f"Data available through: **{data['data_as_of_month']}**  |  "
        f"Forecast target month: **{data['forecast_month']}**"
    )

    st.warning(
        "Historical-data demonstration: the latest available feature "
        "month is October 2025. These results are not live forecasts "
        "for November 2026."
    )

    congested_count = sum(
        result["predicted_congestion"] for result in results
    )

    highest_risk = max(
        results,
        key=lambda item: item["congestion_probability"]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Ports monitored", data["count"])
    c2.metric("Congestion predicted", congested_count)
    c3.metric("No congestion predicted", data["count"] - congested_count)
    c4.metric(
        "Highest risk port",
        format_port(highest_risk["port"]),
        f"{highest_risk['congestion_probability']:.1%}",
        delta_color="inverse"
    )

    st.divider()

    st.subheader("Port Congestion Overview")

    display_df = df[
        [
            "port",
            "congestion_probability",
            "classification_threshold",
            "prediction",
            "port_congestion_threshold_days"
        ]
    ].copy()

    display_df["congestion_probability"] *= 100
    display_df["classification_threshold"] *= 100
    display_df["port"] = display_df["port"].apply(format_port)

    display_df = display_df.rename(columns={
        "port": "Port",
        "congestion_probability": "Congestion Probability (%)",
        "classification_threshold": "Decision Threshold (%)",
        "prediction": "Prediction",
        "port_congestion_threshold_days": "Dwell Threshold (days)"
    })

    st.dataframe(
        display_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Congestion Probability by Port")

    chart_df = df.set_index("port")[
        ["congestion_probability"]
    ].copy()

    chart_df.index = chart_df.index.map(format_port)
    chart_df["congestion_probability"] *= 100
    chart_df.columns = ["Probability (%)"]

    st.bar_chart(chart_df)

    st.divider()
    st.subheader("Individual Port Predictions & Explanations")

    columns = st.columns(2)

    for index, result in enumerate(results):
        with columns[index % 2]:
            port = format_port(result["port"])
            probability = result["congestion_probability"] * 100
            threshold = result["classification_threshold"] * 100
            explanation = result.get("explanation", {})

            st.markdown(f"### {port}")

            if result["predicted_congestion"]:
                st.error("CONGESTION PREDICTED")
            else:
                st.success("NO CONGESTION PREDICTED")

            st.metric(
                "Congestion Probability",
                f"{probability:.2f}%"
            )

            st.progress(
                min(max(probability / 100, 0), 1)
            )

            st.write(f"**Decision threshold:** {threshold:.2f}%")
            st.write(
                f"**Port dwell threshold:** "
                f"{result['port_congestion_threshold_days']:.2f} days"
            )

            st.caption(
                f"Threshold status: {result['threshold_status']}"
            )

            with st.expander("Why did the model make this prediction?"):
                st.write(
                    "**Factors increasing the congestion score**"
                )
                show_factors(
                    explanation.get("top_risk_increasing_factors", []),
                    True
                )

                st.divider()

                st.write(
                    "**Factors decreasing the congestion score**"
                )
                show_factors(
                    explanation.get("top_risk_decreasing_factors", []),
                    False
                )

                st.divider()

                st.write(
                    f"**Model decision score:** "
                    f"{explanation.get('decision_score', 'N/A')}"
                )

                st.caption(
                    explanation.get(
                        "interpretation",
                        "Explanation unavailable."
                    )
                )

    st.divider()

    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model:** Logistic Regression")
        st.write("**Explanation:** Coefficient contributions")
        st.write("**Output space:** Log-odds")

    with col2:
        st.write(f"**Feature month:** {results[0]['feature_month']}")
        st.write(f"**Forecast month:** {data['forecast_month']}")
        st.write("**Ports supported:** 5")

    st.caption(
        "Model contributions describe associations with the model's "
        "congestion score, not causal effects. Probability and decision "
        "threshold are separate: a port is classified as congested when "
        "its probability meets or exceeds its applicable threshold."
    )

except requests.exceptions.ConnectionError:
    st.error(
        "Cannot connect to FastAPI. Start the API server first."
    )

except requests.exceptions.RequestException as exc:
    st.error(f"API request failed: {exc}")

except (KeyError, ValueError, TypeError) as exc:
    st.error(f"Unexpected API response: {exc}")
