# Data Sources and Integration Notes

## 1. Kaggle shipment dataset

File expected at:
`data/raw/global_supply_chain_risk_2026.csv`

Source:
https://www.kaggle.com/datasets/nudratabbas/global-supply-chain-risk-and-logistics-2024-2026

This project uses `Disruption_Occurred` as the shipment-level outcome. `Lead_Time_Days`
is excluded from prediction features until its availability timing is established,
because it may be observed only after a shipment has progressed or completed.

## 2. AIS-derived monthly port dwell data

Files downloaded by `python -m src.download_data`:
- `data/raw/monthly_dwell_2009_2014.csv`
- `data/raw/monthly_dwell_2015_2025.csv`
- Combined: `data/raw/monthly_dwell_2009_2025.csv`

Source:
https://zenodo.org/records/21820262

File:
`monthly_dwell_2015_2025.csv`

Expected MD5:
`6b047f59efe73c69ced561393deee01b`

The dataset contains monthly metrics for five US port complexes:
Houston, LA/Long Beach, NY/NJ, Savannah, and Seattle. It is derived from
terrestrial AIS vessel-position reports. The project uses the processed monthly
summary rather than the multi-gigabyte raw AIS archives.

## 3. Port-name matching limitation

The uploaded Kaggle dataset has nine named ports. Only `Los Angeles` directly
matches the AIS dataset's `LA_Long_Beach` port complex. Consequently:

- The port-congestion model uses all five AIS-covered port complexes.
- The hybrid shipment-disruption model is restricted to shipments with
  Los Angeles as origin or destination.
- No AIS congestion values are fabricated for Antwerp, Busan, Dubai, Hamburg,
  Marseille, Rotterdam, Shanghai, or Singapore.

## 4. Target definitions

The congestion model predicts whether next month's mean vessel dwell time
exceeds the port-specific 75th percentile calculated from the training period
(up to December 2023). The threshold is fitted on training data only.

The shipment model predicts the supplied `Disruption_Occurred` label. It does
not claim that this label is a direct measurement of port congestion.

## 5. Temporal alignment

For a shipment in month M, the hybrid feature join uses AIS observations from
month M-1. This avoids using same-month AIS data that may not have been
available at the shipment's prediction time.


The two port-month files cover five US port complexes. They are concatenated
into one panel after checksum verification. The combined file is used for the
five-port congestion model.

## 3. Uploaded macroeconomic + LA dwell panel

File:
`data/raw/analysis_dataset_dwell.csv`

Zenodo source:
https://zenodo.org/records/21820262

Verified MD5:
`ff45356fa3f9aadad70a1ec75e3f7e67`

This 297-row monthly panel covers January 2001 through September 2025.
Its LA/LB dwell measures begin in January 2009. It contains macroeconomic
indicators (CPI, industrial production, oil prices, NY Fed GSCPI) and LA/LB
dwell/vessel measures. It is not the full five-port panel; use the separate
monthly per-port AIS files for the five-port congestion model.

The v2.0.0 dataset corrected the GSCPI month alignment. Use this uploaded
version rather than an older v1 copy.
