
# PortGuard AI
### Predicting Port Congestion and Shipment Delays Using AI

PortGuard AI is a machine-learning prototype that estimates the likelihood of monthly port congestion across five major US ports using historical vessel activity, dwell-time patterns and supply-chain indicators.

The application provides:
- Congestion probability for each port.
- Port-specific classification thresholds.
- Predicted congestion status.
- Model-based explanations showing factors that increase or decrease the congestion score.
- A Streamlit dashboard and FastAPI prediction service.

## Supported Ports

- Houston
- Los Angeles / Long Beach
- New York / New Jersey
- Savannah
- Seattle

## Architecture

```text
                  Historical Data
                         |
                         v
                Data Preprocessing
                         |
                         v
                 Feature Engineering
                         |
                         v
                Trained ML Model
                         |
             +-----------+-----------+
             |                       |
             v                       v
        FastAPI Backend       Model Explainability
             |                       |
             +-----------+-----------+
                         |
                         v
                 Streamlit Dashboard
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Model | Logistic Regression |
| Backend API | FastAPI |
| API server | Uvicorn |
| Dashboard | Streamlit |
| Model persistence | Joblib |

## Project Structure

```text
PORTGUARD-AI-PROJ/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── five_port_congestion_model.joblib
│
├── reports/
│
├── src/
│   ├── api.py
│   ├── api_schemas.py
│   ├── inference.py
│   ├── port_explanations.py
│   ├── data_preprocessing.py
│   ├── train_port_congestion.py
│   └── explainability.py
│
└── tests/
```

## Model Approach

A Logistic Regression classifier is used to estimate the probability of congestion in the following month.

The model uses a preprocessing pipeline:

1. Median imputation for missing numerical values.
2. Standard scaling of numerical features.
3. One-hot encoding of port identity.
4. Class-weighted Logistic Regression.

### Features

The model uses the following features:

- Unique vessel count
- Mean dwell time
- Median dwell time
- Dwell-time standard deviation
- Total observations
- Average vessel length
- Three-month rolling dwell time
- Monthly dwell-time change
- Monthly vessel-count change
- Global Supply Chain Pressure Index (GSCPI)
- Calendar month
- Port identity

### Prediction Logic

The model produces a congestion probability for each port.

The probability is compared against the applicable classification threshold to determine the final prediction.

Port-specific dwell thresholds are used in defining the congestion target.

## Model Evaluation

Reported held-out evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 79.51% |
| Precision | 71.43% |
| Recall | 98.04% |
| F1 Score | 82.64% |
| F2 Score | 91.24% |
| ROC-AUC | 0.9371 |
| PR-AUC | 0.9346 |
| Brier Score | 0.1255 |

Confusion matrix:

```text
                 Predicted
                 No     Yes

Actual No        63      40
Actual Yes        2     100
```

## Explainability

PortGuard AI provides prediction-level explanations using Logistic Regression coefficient contributions.

For each port, the application displays:
- Top factors increasing the congestion score.
- Top factors decreasing the congestion score.
- Model decision score.

These contributions explain the model's associations with its prediction. They do not establish causal relationships.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PORTGUARD-AI-PROJ
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

```bash
python -m uvicorn src.api:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### 5. Start the Streamlit dashboard

Open another terminal:

```bash
streamlit run app.py
```

Dashboard:

http://localhost:8501

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/ports` | Supported ports |
| POST | `/predict` | Predict for one port |
| POST | `/predict/batch` | Batch predictions |
| GET | `/predict/all-ports` | Predictions for all supported ports |

## Data Availability and Limitations

The current inference run uses features through October 2025 and produces predictions for November 2025.

These are historical-data demonstration results, not live forecasts for November 2026.

The model currently covers five US ports. Prediction quality depends on the availability, completeness and timing of input data.

Further validation using updated operational data, observed queue lengths and vessel turnaround outcomes is recommended before operational deployment.

## Future Improvements

- Integrate regularly refreshed AIS and macroeconomic data.
- Add more ports after validating their data and congestion definitions.
- Perform rolling time-based model validation.
- Monitor model calibration and prediction drift.
- Incorporate additional operational indicators such as queue lengths and terminal waiting times.
- Deploy the API and dashboard to a cloud environment.

## Project Deliverables

- Machine-learning model and inference pipeline.
- FastAPI prediction service.
- Streamlit dashboard.
- Prediction-level explainability.
- Technical report.

---

**PortGuard AI**  
AI-based port congestion prediction prototype.
