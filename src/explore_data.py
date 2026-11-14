from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"

def save_line(data, x, y, title, xlabel, ylabel, path, group=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    if group:
        for name, part in data.groupby(group):
            ax.plot(part[x], part[y], label=str(name), linewidth=1.3)
        ax.legend(ncol=2)
    else:
        ax.plot(data[x], data[y], linewidth=1.5)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    macro_path = RAW / "analysis_dataset_dwell.csv"
    if not macro_path.exists():
        raise FileNotFoundError(f"Missing {macro_path}")
    macro = pd.read_csv(macro_path, parse_dates=["date"])

    summary = {
        "macro_panel_rows": int(len(macro)),
        "macro_panel_columns": int(macro.shape[1]),
        "macro_date_start": str(macro.date.min().date()),
        "macro_date_end": str(macro.date.max().date()),
        "macro_duplicate_rows": int(macro.duplicated().sum()),
        "macro_missing_values": {k: int(v) for k, v in macro.isna().sum().items()},
        "la_dwell_coverage_start": str(macro.loc[macro.dwell_la_raw.notna(), "date"].min().date()),
        "la_dwell_coverage_end": str(macro.loc[macro.dwell_la_raw.notna(), "date"].max().date()),
        "la_dwell_gscpi_pearson": float(macro[["dwell_la_raw", "gscpi"]].corr().iloc[0, 1]),
    }
    save_line(
        macro.dropna(subset=["dwell_la_raw"]), "date", "dwell_la_raw",
        "LA / Long Beach Monthly Mean Vessel Dwell", "Month", "Mean dwell (days)",
        FIG / "la_dwell_trend.png"
    )

    valid = macro[["gscpi", "dwell_la_raw"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid.gscpi, valid.dwell_la_raw, alpha=0.65)
    ax.set(title="LA/LB Dwell vs Global Supply Chain Pressure",
           xlabel="GSCPI", ylabel="Mean dwell (days)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "dwell_vs_gscpi_scatter.png", dpi=160)
    plt.close(fig)

    port_path = RAW / "monthly_dwell_2009_2025.csv"
    if port_path.exists():
        port = pd.read_csv(port_path, parse_dates=["month"])
        summary["port_panel_rows"] = int(len(port))
        summary["ports"] = sorted(port.Port.dropna().unique().tolist())
        summary["port_panel_month_start"] = str(port.month.min().date())
        summary["port_panel_month_end"] = str(port.month.max().date())
        summary["port_month_duplicates"] = int(port.duplicated(["Port", "month"]).sum())
        save_line(
            port.sort_values("month"), "month", "MeanDwellDays",
            "Monthly Mean Vessel Dwell by Port", "Month", "Mean dwell (days)",
            FIG / "five_port_dwell_trends.png", group="Port"
        )
        summary["mean_dwell_by_port"] = (
            port.groupby("Port").MeanDwellDays.mean().round(3).to_dict()
        )
    else:
        summary["port_panel_note"] = "Run python -m src.download_data to fetch five-port files."

    (OUT / "phase3_eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
