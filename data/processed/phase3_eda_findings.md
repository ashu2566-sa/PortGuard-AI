# Phase 3 – Initial EDA Findings

## Dataset identity

- File: `analysis_dataset_dwell.csv`
- Verified source: Zenodo record 21820262, version 2.0.0
- MD5: `ff45356fa3f9aadad70a1ec75e3f7e67`
- Records: 297
- Columns: 16
- Full date range: 2001-01 to 2025-09
- LA/LB dwell coverage: 2009-01 to 2025-09
- Dwell observations: 201
- Duplicate rows: 0
- Duplicate dates: 0

## Data quality

Missing values are concentrated in the AIS dwell columns before January 2009:
- `dwell_composite`: 96 missing
- `dwell_la_raw`: 96 missing
- `la_unique_vessels`: 96 missing
- `dwell_la_detrended`: 96 missing

`oil_price` and `oil_yoy` each contain 3 missing values. These will be handled explicitly during feature preparation.

## Initial statistical observations

- LA/LB raw dwell mean: 4.936 days.
- LA/LB raw dwell median: 4.866 days.
- LA/LB raw dwell standard deviation: 0.884 days.
- Unique vessels mean: 308.5.
- Correlation between raw LA dwell and GSCPI: 0.586.
- Correlation between detrended LA dwell and GSCPI: 0.508.

Correlations are descriptive associations, not causal evidence.

## Modelling implications

1. This panel is monthly and LA/LB-specific for the dwell fields; it is not itself the full five-port panel.
2. The companion `monthly_dwell_2009_2014.csv` and `monthly_dwell_2015_2025.csv` files provide port-by-month metrics for all five port complexes.
3. Use lagged dwell and lagged macroeconomic variables when predicting future congestion. Never use same-month/future observations if they would not be available at prediction time.
4. Treat `dwell_composite` and `dwell_la_detrended` as derived measures until their exact construction is documented from the dataset's companion methodology. Do not silently assume they are raw dwell days.
5. The Kaggle shipment outcome `Disruption_Occurred` is a separate target. It is not a direct label for port congestion.
6. Only the Los Angeles shipment subset can be directly joined to the LA/LB AIS observations. Do not impute AIS measurements for other Kaggle ports.

## Generated figures

- `reports/figures/la_dwell_and_vessel_counts.png`
- `reports/figures/detrended_dwell_vs_gscpi.png`
- `reports/figures/dwell_vs_gscpi_scatter.png`
- `reports/figures/la_dwell_monthly_seasonality.png`
