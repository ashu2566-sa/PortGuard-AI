from pathlib import Path
import hashlib
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FILES = {
    "monthly_dwell_2009_2014.csv": {
        "url": "https://zenodo.org/records/21820262/files/monthly_dwell_2009_2014.csv?download=1",
        "md5": "4e8e03d28fb49c9df9a6fa6d6d47fbe8",
    },
    "monthly_dwell_2015_2025.csv": {
        "url": "https://zenodo.org/records/21820262/files/monthly_dwell_2015_2025.csv?download=1",
        "md5": "6b047f59efe73c69ced561393deee01b",
    },
}

def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_file(name: str, metadata: dict) -> Path:
    path = RAW / name
    if path.exists():
        actual = md5(path)
        if actual == metadata["md5"]:
            print(f"Verified existing file: {name}")
            return path
        raise ValueError(
            f"Checksum mismatch for existing {name}: {actual}. "
            "Remove the file and rerun."
        )

    response = requests.get(metadata["url"], timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)
    actual = md5(path)

    if actual != metadata["md5"]:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {name}: expected {metadata['md5']}, "
            f"received {actual}. Verify the Zenodo record."
        )

    print(f"Downloaded and verified: {name}")
    return path

def combine_dwell_files() -> Path:
    frames = [__import__("pandas").read_csv(RAW / name) for name in FILES]
    import pandas as pd
    combined = pd.concat(frames, ignore_index=True)
    combined["month"] = pd.to_datetime(
        combined["YearMonth"].astype(str) + "-01",
        errors="coerce"
    )
    combined = combined.dropna(subset=["month", "Port"])
    combined = combined.sort_values(["Port", "month"])
    if combined.duplicated(["Port", "month"]).any():
        raise ValueError("Duplicate Port × month records found after combining files.")
    output = RAW / "monthly_dwell_2009_2025.csv"
    combined.to_csv(output, index=False)
    print(f"Combined {len(combined):,} port-month rows: {output}")
    return output

def download_ais_data() -> Path:
    for name, metadata in FILES.items():
        download_file(name, metadata)
    return combine_dwell_files()

if __name__ == "__main__":
    download_ais_data()
