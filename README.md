# FAILSAFE — Student At-Risk Early Warning System

A full-stack, AI-powered web application designed to identify students at risk of academic failure before end-of-semester results are released, explain the factors behind each prediction using Explainable AI, and generate personalised intervention plans for faculty.

---

## The Problem

In educational institutions, students at risk of failing are often identified only after final grades are released, leaving limited opportunity for meaningful intervention. Faculty need a proactive, data-driven system that can identify struggling students early while also providing insight into the factors contributing to their academic risk.

## The Solution

FAILSAFE enables faculty to upload student datasets, obtain instant risk predictions with transparent SHAP-based explanations, analyse cohort-level trends, and generate AI-assisted intervention plans — allowing action to be taken before final outcomes are determined.

---

## Key Metrics

| Metric                         | Value        |
| ------------------------------ | ------------ |
| CV ROC-AUC                     | 0.985        |
| Test ROC-AUC                   | 0.970        |
| At-Risk Recall                 | 92.3%        |
| Accuracy                       | 89.9%        |
| At-Risk Precision              | 80.0%        |
| Batch Processing (10k records) | ~2.1 seconds |

---

## Features

* **Batch Prediction Pipeline** — Upload a CSV and process the entire cohort through vectorised ML inference, SHAP analysis, and rule-based intervention flags, followed by a single bulk database insert.
* **Explainable AI** — Per-student SHAP visualisations highlight the features that contribute most strongly to each risk prediction.
* **AI-Generated Intervention Plans** — Gemini 2.5 Flash generates personalised, structured intervention plans for flagged students using their risk score, SHAP drivers, and intervention flags.
* **Intervention Tracking** — Faculty can mark students as actioned, with timestamps and usernames recorded for auditability.
* **Cohort Analytics** — Analyse batch-level risk distributions and identify the most influential risk drivers across an entire cohort.
* **Batch History** — Maintains a complete history of previous prediction runs in PostgreSQL for tracking and auditing.
* **JWT Authentication** — Secure authentication and session management with automatic logout when tokens expire.

---

## Tech Stack

| Layer            | Technologies                                                                 |
| ---------------- | ---------------------------------------------------------------------------- |
| Machine Learning | Python, XGBoost, scikit-learn, SHAP, SMOTE (imbalanced-learn), Pandas, NumPy |
| Backend          | FastAPI, SQLAlchemy, PostgreSQL, Google Gemini 2.5 Flash, JWT Auth           |
| Frontend         | React.js, Tailwind CSS, Recharts, Axios                                      |
| Infrastructure   | Docker, Docker Compose                                                       |

---

## Project Structure

```text
FailSafe/
├── data/                   # Dataset (not tracked — see Setup)
├── models/                 # Trained model artifacts (included in repo)
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

* Docker and Docker Compose
* Python 3.10+
* Gemini API Key — available from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/Harish18010/FailSafe.git
cd FailSafe
```

### 2. Download the Dataset

Download the UCI Student Performance dataset from Kaggle and place it at:

```text
data/student_data.csv
```

Dataset:

https://www.kaggle.com/datasets/larsen0966/student-performance-data-set

### 3. Train the Model (Optional)

Pre-trained model artifacts are already included in the `models/` directory. Retraining is only required if you want to experiment with the training pipeline or different hyperparameters.

```bash
cd src/ml
pip install -r ../api/requirements.txt
python trainer.py
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_random_secret_key_here
```

Generate a secure secret key using:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 5. Run with Docker

```bash
docker-compose up -d --build
```

After the first startup, run the database migration once:

```bash
docker-compose exec db psql -U failsafe_admin -d failsafe_production -c "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actioned BOOLEAN DEFAULT FALSE; ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actioned_by VARCHAR; ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actioned_at TIMESTAMPTZ; ALTER TABLE predictions ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();"
```

### 6. Access the Application

Once the Docker containers are running:

* **Frontend:** `http://localhost:5173`
* **API Documentation:** `http://localhost:8000/docs`

**Default credentials:** `faculty_admin` / `securepassword123`

---

## API Endpoints

| Method | Endpoint                         | Description                               |
| ------ | -------------------------------- | ----------------------------------------- |
| POST   | `/token`                         | Authenticate and obtain a JWT token       |
| GET    | `/health`                        | Check API and model health                |
| POST   | `/predict/batch`                 | Upload a CSV and run batch predictions    |
| GET    | `/predict/history`               | Retrieve previous prediction records      |
| POST   | `/predict/action/{student_id}`   | Mark a student intervention as actioned   |
| POST   | `/predict/unaction/{student_id}` | Reverse a student's action status         |
| POST   | `/agent/draft-intervention`      | Generate an AI-assisted intervention plan |

---

## How It Works

**Data Loading & Feature Engineering** — Grade-trend features such as `G2-G1` and the mid-term average are engineered before encoding. The `OrdinalEncoder` is fit exclusively on the training data to prevent data leakage.

**Model Training** — XGBoost is trained with SMOTE oversampling to address class imbalance. Model selection uses manual 5-fold cross-validation across 60 randomly sampled hyperparameter combinations, followed by data-driven decision-threshold calibration using Youden's J statistic.

**Batch Inference** — Uploaded CSV files are processed using the same preprocessing pipeline used during training. Risk probabilities are computed through vectorised `predict_proba` calls, while SHAP values are calculated in batched NumPy operations. Prediction records are persisted using SQLAlchemy's `bulk_insert_mappings`. The pipeline was benchmarked at approximately **2.1 seconds for 10,270 student records**.

**Explainability** — `shap.TreeExplainer` computes feature-level contributions for every student prediction. The five strongest SHAP drivers are returned for each student and displayed through colour-coded contribution charts.

**Intervention Generation** — For students identified as at risk, Gemini 2.5 Flash receives the student's predicted risk score, most influential SHAP drivers, and rule-based intervention flags to produce a structured three-part intervention plan.

---

## Workflow

```text
Student CSV
    │
    ▼
Preprocessing & Feature Engineering
    │
    ▼
XGBoost Risk Prediction
    │
    ├──► SHAP Feature Attribution
    │
    ├──► Rule-Based Intervention Flags
    │
    ▼
PostgreSQL Prediction History
    │
    ▼
Faculty Dashboard
    │
    ├──► Cohort Analytics
    ├──► Student-Level Explanations
    ├──► Intervention Tracking
    └──► Gemini-Generated Intervention Plans
```

---

*Built for educational purposes.*
