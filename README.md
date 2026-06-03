# FAILSAFE — Student At-Risk Early Warning System

A full-stack, AI-powered web application that predicts student failure risk before end-of-semester results, explains predictions using Explainable AI, and auto-generates personalised intervention plans for faculty.

---

## The Problem

In educational institutions, student failure often goes undetected until final grades are released — leaving no room for meaningful intervention. Faculty lack a proactive, data-driven tool to identify struggling students early and understand the root causes behind their performance.

## The Solution

FAILSAFE allows faculty to upload student datasets, get instant at-risk predictions with transparent explanations (SHAP values), and generate AI-powered intervention plans — all before it's too late.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC Score | 0.970 |
| At-Risk Recall | 92.3% |
| Accuracy | 89.9% |
| At-Risk Precision | 80.0% |

---

## Features

- **Batch Prediction Pipeline** — Upload a CSV of student records; the system runs vectorised ML inference, SHAP analysis, and rule-based intervention flags in a single pass with bulk DB insert
- **Explainable AI** — Per-student SHAP waterfall charts show exactly which features drive the risk prediction
- **AI Intervention Plans** — Gemini 2.5 Flash generates personalised, structured intervention plans per flagged student based on their SHAP risk drivers
- **Intervention Tracking** — Faculty can mark students as "Actioned" with timestamp and username audit trail
- **Cohort Analytics** — Batch-level risk distribution, top risk drivers across the cohort, and model metrics
- **Batch History** — Full audit trail of all past prediction runs stored in PostgreSQL
- **JWT Authentication** — Secure session management with auto-logout on token expiry

---

## Tech Stack

**Machine Learning**
- Python, XGBoost, scikit-learn, SHAP, SMOTE (imbalanced-learn)
- Pandas, NumPy, Matplotlib

**Backend**
- FastAPI, SQLAlchemy, PostgreSQL
- Google Gemini 2.5 Flash API
- JWT Authentication (python-jose, passlib)

**Frontend**
- React.js, Tailwind CSS, Recharts
- Axios

**Infrastructure**
- Docker, Docker Compose

---

## Project Structure

```
FailSafe/
├── data/                   # Dataset (not tracked — see below)
├── models/                 # Trained model artifacts (not tracked)
├── src/
│   ├── api/                # FastAPI backend
│   │   ├── main.py
│   │   ├── auth.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── intervention.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── ml/                 # ML training pipeline
│       ├── trainer.py
│       ├── data_loader.py
│       └── explainer.py
├── frontend/               # React frontend
├── docker-compose.yml
└── .env
```

---

## Setup & Installation

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+
- Gemini API Key (free at [aistudio.google.com](https://aistudio.google.com/app/apikey))

### 1. Clone the repository

```bash
git clone https://github.com/Harish18010/FailSafe.git
cd FailSafe
```

### 2. Download the dataset

Download the UCI Student Performance dataset from Kaggle:
[https://www.kaggle.com/datasets/devansodariya/student-performance-data](https://www.kaggle.com/datasets/devansodariya/student-performance-data)

Place it at: `data/student_data.csv`

### 3. Train the model

```bash
cd src/ml
pip install -r ../api/requirements.txt
python trainer.py
```

This generates `models/failsafe_xgb.json`, `failsafe_xgb_meta.pkl`, and `failsafe_xgb_encoder.pkl`.

### 4. Configure environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_random_secret_key_here
```

Generate a secret key with:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 5. Run with Docker

```bash
docker-compose up -d --build
```

Run the DB migration for tracking columns:
```bash
docker-compose exec db psql -U failsafe_admin -d failsafe_production -c "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actioned BOOLEAN DEFAULT FALSE; ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actioned_by VARCHAR; ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actioned_at TIMESTAMPTZ; ALTER TABLE predictions ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();"
```

### 6. Access the app

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

**Default credentials:** `faculty_admin` / `securepassword123`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/token` | Login and get JWT token |
| GET | `/health` | Model and API health check |
| POST | `/predict/batch` | Upload CSV and run batch prediction |
| GET | `/predict/history` | Get all past prediction records |
| POST | `/predict/action/{student_id}` | Mark student intervention as actioned |
| POST | `/predict/unaction/{student_id}` | Reverse action status |
| POST | `/agent/draft-intervention` | Generate AI intervention plan |

---

## How It Works

1. **Data Loading** — Grade trend features (`G2-G1`, mid-term average) are engineered before encoding. `OrdinalEncoder` is fit only on training data to prevent leakage.

2. **Training** — XGBoost with SMOTE oversampling, manual 5-fold CV across 60 random hyperparameter combinations, and data-driven threshold calibration via Youden's J statistic.

3. **Inference** — Uploaded CSV is preprocessed identically to training data using the saved encoder. Vectorised `predict_proba` + SHAP in single numpy passes. Bulk DB insert via `bulk_insert_mappings`.

4. **Explainability** — `shap.TreeExplainer` computes per-student feature contributions. Top 5 SHAP drivers are returned per student and rendered as a colour-coded bar chart.

5. **Intervention** — Gemini 2.5 Flash receives the student's SHAP drivers, risk score, and rule-based flags to generate a structured 3-part intervention plan (root cause, immediate actions, monitoring plan).

---

## Resume Metrics

- Achieved **97% ROC-AUC** and **92% at-risk recall** using XGBoost + SMOTE + engineered grade trajectory features on UCI Student Performance dataset
- Architected a **vectorised FastAPI batch pipeline** capable of processing 10,000+ student records with single-pass SHAP computation and bulk PostgreSQL insert
- Integrated **Gemini 2.5 Flash** to auto-generate personalised faculty intervention plans, eliminating ~4 hours/student of manual planning per week

---

## License

This project is done for educational purposes only.