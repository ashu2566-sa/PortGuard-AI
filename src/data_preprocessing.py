from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

KAGGLE_FILE = RAW / "global_supply_chain_risk_2026.csv"
AIS_FILE = RAW / "monthly_dwell_2009_2025.csv"
MACRO_FILE = RAW / "analysis_dataset_dwell.csv"

PORT_ALIASES = {
    "Los Angeles": "LA_Long_Beach",
}

AIS_NUMERIC = [
    "UniqueVessels", "MeanDwellDays", "MedianDwellDays",
    "StdDwellDays", "MinDwellDays", "MaxDwellDays",
    "TotalObservations", "AvgVesselLength", "days_ok", "days_total",
]

def load_sources():
    if not KAGGLE_FILE.exists():
        raise FileNotFoundError(
            f"Place the Kaggle CSV at {KAGGLE_FILE}"
        )
    if not AIS_FILE.exists():
        raise FileNotFoundError(
            f"Run `python -m src.download_data` first. Missing {AIS_FILE}"
        )

    shipments = pd.read_csv(KAGGLE_FILE)
    ais = pd.read_csv(AIS_FILE)
    if not MACRO_FILE.exists():
        raise FileNotFoundError(f"Missing uploaded macro/AIS panel: {MACRO_FILE}")
    macro = pd.read_csv(MACRO_FILE)
    return shipments, ais, macro

def clean_ais(ais: pd.DataFrame) -> pd.DataFrame:
    required = {"Port", "YearMonth", *AIS_NUMERIC}
    missing = required.difference(ais.columns)
    if missing:
        raise ValueError(f"AIS file is missing columns: {sorted(missing)}")

    result = ais.copy()
    result["month"] = pd.to_datetime(
        result["YearMonth"].astype(str) + "-01",
        errors="coerce"
    )
    result = result.dropna(subset=["month", "Port"])
    for column in AIS_NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.sort_values(["Port", "month"])
    result = result.drop_duplicates(["Port", "month"], keep="last")
    return result

def build_congestion_panel(ais: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create a next-month congestion target using past-only predictors."""
    panel = clean_ais(ais)
    if macro is not None:
        macro = macro.copy()
        macro["month"] = pd.to_datetime(macro["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        macro_columns = ["month", "gscpi", "cpi_yoy", "cpi_goods_yoy",
                         "cpi_services_yoy", "indpro_yoy", "oil_yoy"]
        macro = macro[macro_columns].drop_duplicates("month")
        panel = panel.merge(macro, on="month", how="left")
    panel = panel.sort_values(["Port", "month"]).copy()

    grouped = panel.groupby("Port", group_keys=False)
    lag_columns = [
        "MeanDwellDays", "MedianDwellDays", "StdDwellDays",
        "UniqueVessels", "TotalObservations", "AvgVesselLength",
        "gscpi", "cpi_yoy", "cpi_goods_yoy", "cpi_services_yoy",
        "indpro_yoy", "oil_yoy",
    ]

    for column in lag_columns:
        # Values observed by the end of month t are used to predict month t+1.
        panel[f"{column}_asof_month"] = grouped[column].shift(0)
        panel[f"{column}_prior_month"] = grouped[column].shift(1)

    panel["dwell_rolling_3m"] = (
        panel.groupby("Port")["MeanDwellDays"]
        .transform(lambda s: s.rolling(3, min_periods=2).mean())
    )
    panel["vessel_count_change"] = (
        panel.groupby("Port")["UniqueVessels"].pct_change()
    )
    panel["month_num"] = panel["month"].dt.month
    panel["year"] = panel["month"].dt.year

    # The target is next month's dwell. It is deliberately shifted forward.
    panel["target_next_month_dwell"] = (
        panel.groupby("Port")["MeanDwellDays"].shift(-1)
    )
    panel["target_month"] = (
        panel.groupby("Port")["month"].shift(-1)
    )

    # Current-month predictors are known at month end; target is next month.
    panel["target_next_month"] = panel.groupby("Port")[
        "target_next_month_dwell"
    ].transform(lambda s: s)

    return panel

def add_congestion_labels(
    panel: pd.DataFrame,
    train_end: str = "2023-12-01",
    quantile: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-port thresholds on training-period target values only."""
    result = panel.copy()
    cutoff = pd.Timestamp(train_end)
    # Thresholds are based only on target months available by the cutoff.
    training = result[result["target_month"] <= cutoff]

    thresholds = (
        training.groupby("Port")["target_next_month_dwell"]
        .quantile(quantile)
        .rename("congestion_threshold_days")
        .reset_index()
    )
    result = result.merge(thresholds, on="Port", how="left")
    result["congestion_label"] = (
        result["target_next_month_dwell"]
        > result["congestion_threshold_days"]
    ).astype("Int64")

    result = result.dropna(
        subset=["target_next_month_dwell", "congestion_threshold_days"]
    )
    return result, thresholds

def build_los_angeles_hybrid_shipments(
    shipments: pd.DataFrame,
    ais: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Join only LA-related shipments to the LA/Long Beach AIS complex."""
    data = shipments.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["month"] = data["Date"].dt.to_period("M").dt.to_timestamp()

    data["origin_ais_port"] = data["Origin_Port"].map(PORT_ALIASES)
    data["destination_ais_port"] = data["Destination_Port"].map(PORT_ALIASES)

    ais_clean = clean_ais(ais)
    la = ais_clean.loc[
        ais_clean["Port"].eq("LA_Long_Beach"),
        ["month", "MeanDwellDays", "MedianDwellDays",
         "UniqueVessels", "StdDwellDays", "TotalObservations"]
    ].sort_values("month").copy()

    # Shift AIS month forward by one month: month M shipments receive
    # month M-1 AIS observations, avoiding same-month look-ahead.
    la["month"] = la["month"] + pd.offsets.MonthBegin(1)
    la = la.rename(columns={
        "MeanDwellDays": "la_mean_dwell_lag1",
        "MedianDwellDays": "la_median_dwell_lag1",
        "UniqueVessels": "la_unique_vessels_lag1",
        "StdDwellDays": "la_dwell_std_lag1",
        "TotalObservations": "la_ais_observations_lag1",
    })

    data = data.merge(la, on="month", how="left")

    macro = macro.copy()
    macro["month"] = pd.to_datetime(macro["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    macro_columns = ["month", "gscpi", "cpi_yoy", "cpi_goods_yoy",
                     "cpi_services_yoy", "indpro_yoy", "oil_yoy"]
    macro = macro[macro_columns].drop_duplicates("month").sort_values("month")
    macro["month"] = macro["month"] + pd.offsets.MonthBegin(1)
    macro = macro.rename(columns={
        "gscpi": "gscpi_lag1",
        "cpi_yoy": "cpi_yoy_lag1",
        "cpi_goods_yoy": "cpi_goods_yoy_lag1",
        "cpi_services_yoy": "cpi_services_yoy_lag1",
        "indpro_yoy": "indpro_yoy_lag1",
        "oil_yoy": "oil_yoy_lag1",
    })
    data = data.merge(macro, on="month", how="left")

    data["is_los_angeles_related"] = (
        data["Origin_Port"].eq("Los Angeles")
        | data["Destination_Port"].eq("Los Angeles")
    ).astype(int)

    # Only retain the cohort for which external AIS coverage exists.
    hybrid = data.loc[data["is_los_angeles_related"].eq(1)].copy()
    hybrid["month_num"] = hybrid["Date"].dt.month
    hybrid["year"] = hybrid["Date"].dt.year
    return hybrid

def run():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    shipments, ais, macro = load_sources()

    panel = build_congestion_panel(ais, macro)
    panel, thresholds = add_congestion_labels(panel)
    hybrid = build_los_angeles_hybrid_shipments(shipments, ais, macro)

    panel.to_csv(PROCESSED / "port_congestion_panel.csv", index=False)
    thresholds.to_csv(PROCESSED / "port_congestion_thresholds.csv", index=False)
    hybrid.to_csv(PROCESSED / "los_angeles_hybrid_shipments.csv", index=False)

    print(f"Congestion panel rows: {len(panel):,}")
    print(f"Hybrid LA shipment rows: {len(hybrid):,}")
    print(f"Ports in AIS data: {sorted(panel['Port'].unique())}")
    print("Saved processed datasets under data/processed/")

if __name__ == "__main__":
    run()
