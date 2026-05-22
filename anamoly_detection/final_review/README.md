# AnomalyIQ — Isolation Forest Web App

A local web application that wraps your Isolation Forest anomaly detection code into a clean REST API + browser UI.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py
```

Then open your browser at → **http://localhost:5000**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Check if model is trained |
| POST | `/api/train` | Upload Train_data.csv to train the model |
| POST | `/api/predict` | Upload Test_data.csv to get anomaly predictions |

### POST /api/train
- **Body**: `multipart/form-data`
  - `file` — CSV file with a `class` column
  - `contamination` — float (default `0.1`)

### POST /api/predict
- **Body**: `multipart/form-data`
  - `file` — CSV file (the `class` column is optional)

### Example with curl
```bash
# Train
curl -X POST http://localhost:5000/api/train \
  -F "file=@Train_data.csv" \
  -F "contamination=0.1"

# Predict
curl -X POST http://localhost:5000/api/predict \
  -F "file=@Test_data.csv"
```

---

## Project Structure
```
anomaly_app/
├── app.py              ← Flask API server
├── requirements.txt
├── README.md
└── static/
    └── index.html      ← Browser UI (served at /)
```
