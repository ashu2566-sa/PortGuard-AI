
from pathlib import Path

import pandas as pd

from src.train_port_congestion import load_port_panel


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

AIS_PATH = RAW / "monthly_dwell_2009_2025.csv"
MACRO_PATH = RAW / "analysis_dataset_dwell.csv"


def build_latest_features():
    """Build the latest complete feature row for each supported port."""

    panel = load_port_panel(AIS_PATH)
    panel = panel.sort_values(["Port", "month"]).copy()

    grouped = panel.groupby("Port", group_keys=False)

    panel["rolling_3m_dwell"] = grouped["MeanDwellDays"].transform(
        lambda values: values.rolling(3, min_periods=2).mean()
    )

    panel["dwell_change_1m"] = grouped["MeanDwellDays"].diff()
    panel["vessel_change_1m"] = grouped["UniqueVessels"].pct_change()
    panel["month_num"] = panel["month"].dt.month

    if not MACRO_PATH.exists():
        raise FileNotFoundError(f"Missing GSCPI data: {MACRO_PATH}")

    macro = pd.read_csv(MACRO_PATH)

    if not {"date", "gscpi"}.issubset(macro.columns):
        raise ValueError("Macro file must contain date and gscpi columns.")

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

    # Match the training pipeline's GSCPI publication lag.
    macro["month"] = macro["month"] + pd.offsets.MonthBegin(1)
    macro = macro[["month", "gscpi"]]

    panel = panel.merge(
        macro,
        on="month",
        how="left",
        validate="many_to_one",
    )

    # Use only months with all required feature inputs available.
    feature_columns = [
        "UniqueVessels",
        "MeanDwellDays",
        "MedianDwellDays",
        "StdDwellDays",
        "TotalObservations",
        "AvgVesselLength",
        "rolling_3m_dwell",
        "dwell_change_1m",
        "vessel_change_1m",
        "gscpi",
        "month_num",
    ]

    panel = panel.dropna(subset=feature_columns).copy()

    if panel.empty:
        raise ValueError("No complete feature rows are available.")

    # Use the latest common month across all five ports.
    latest_month = panel.groupby("Port")["month"].max().min()

    latest = panel.loc[panel["month"].eq(latest_month)].copy()

    expected_ports = {
        "Houston",
        "LA_Long_Beach",
        "NY_NJ",
        "Savannah",
        "Seattle",
    }

    available_ports = set(latest["Port"].unique())
    missing_ports = expected_ports - available_ports

    if missing_ports:
        raise ValueError(
            f"Missing latest feature data for ports: {sorted(missing_ports)}"
        )

    latest = latest.drop_duplicates("Port", keep="last")

    return latest.sort_values("Port").reset_index(drop=True)
