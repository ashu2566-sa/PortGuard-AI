from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    port: str
    feature_month: str = Field(
        description="Month for which the input features are available, in YYYY-MM format"
    )

    unique_vessels: float = Field(ge=0)
    mean_dwell_days: float = Field(ge=0)
    median_dwell_days: float = Field(ge=0)
    std_dwell_days: float = Field(ge=0)
    total_observations: float = Field(ge=0)
    avg_vessel_length: float = Field(ge=0)

    rolling_3m_dwell: float = Field(ge=0)
    dwell_change_1m: float
    vessel_change_1m: float
    gscpi: float
    month_num: int = Field(ge=1, le=12)


class BatchPredictionInput(BaseModel):
    predictions: list[PredictionInput] = Field(min_length=1, max_length=100)