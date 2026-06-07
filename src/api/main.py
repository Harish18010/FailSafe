from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import xgboost as xgb
import joblib
import shap
import numpy as np
import pandas as pd
import io
import os
import uuid
import time
import pathlib
from datetime import datetime, timezone

from database import get_db, engine, Base
from models import PredictionRecord, InterventionRule, User
from auth import auth_router, get_current_active_user, get_password_hash
from intervention import generate_intervention

app = FastAPI(title="FAILSAFE API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

_DOCKER_MODELS = pathlib.Path("/app/models")
_LOCAL_MODELS  = pathlib.Path(__file__).resolve().parent.parent.parent / "models"
_MODELS_DIR    = _DOCKER_MODELS if _DOCKER_MODELS.exists() else _LOCAL_MODELS

MODEL_PATH   = str(_MODELS_DIR / "failsafe_xgb.pkl")
META_PATH    = str(_MODELS_DIR / "failsafe_xgb_meta.pkl")
ENCODER_PATH = str(_MODELS_DIR / "failsafe_xgb_encoder.pkl")

ml_components = {}


@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)

    db_gen = get_db()
    db = next(db_gen)
    try:
        if not db.query(User).filter(User.username == "faculty_admin").first():
            hashed_pw = get_password_hash("securepassword123")
            admin_user = User(username="faculty_admin", hashed_password=hashed_pw, is_active=True)
            db.add(admin_user)
            db.commit()

        if db.query(InterventionRule).count() == 0:
            default_rules = [
                InterventionRule(feature="absences",  operator=">", threshold=10, plan_text="High absences: Trigger attendance counselor workflow."),
                InterventionRule(feature="failures",  operator=">", threshold=0,  plan_text="Past failures: Assign to mandatory peer tutoring."),
                InterventionRule(feature="studytime", operator="<", threshold=2,  plan_text="Low study time: Provide time-management workshop."),
            ]
            db.add_all(default_rules)
            db.commit()
    finally:
        db.close()

    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return

    json_path = MODEL_PATH.replace('.pkl', '.json')
    if os.path.exists(json_path):
        model = xgb.XGBClassifier()
        model.load_model(json_path)
    else:
        model = joblib.load(MODEL_PATH)
    ml_components["model"] = model

    meta = joblib.load(META_PATH)
    ml_components["features"]  = meta["feature_names"]
    ml_components["threshold"] = meta["threshold"]
    ml_components["accuracy"]  = meta.get("cv_roc_auc", 0)
    ml_components["explainer"] = None

    if os.path.exists(ENCODER_PATH):
        enc_bundle = joblib.load(ENCODER_PATH)
        ml_components["encoder"]  = enc_bundle["encoder"]
        ml_components["cat_cols"] = enc_bundle["cat_cols"]


@app.get("/health")
async def health_check():
    return {
        "status":         "healthy",
        "model_loaded":   ml_components.get("model") is not None,
        "threshold":      ml_components.get("threshold"),
        "model_accuracy": ml_components.get("accuracy"),
    }


def _get_explainer():
    if ml_components.get("explainer") is None:
        ml_components["explainer"] = shap.TreeExplainer(ml_components["model"])
    return ml_components["explainer"]


def _preprocess_uploaded_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "G1" in df.columns and "G2" in df.columns:
        df["grade_trend"] = df["G2"] - df["G1"]
        df["mid_avg"]     = (df["G1"] + df["G2"]) / 2
    drop_cols = ["G1", "G2", "G3", "at_risk"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if "encoder" in ml_components and "cat_cols" in ml_components:
        cat_cols = [c for c in ml_components["cat_cols"] if c in df.columns]
        if cat_cols:
            df[cat_cols] = ml_components["encoder"].transform(df[cat_cols])
    features = ml_components["features"]
    df = df[[c for c in features if c in df.columns]]
    return df


def _record_to_dict(r):
    return {
        "batch_id":          r.batch_id,
        "student_id":        r.student_id,
        "risk_probability":  r.risk_probability,
        "is_at_risk":        r.is_at_risk,
        "intervention_plan": r.intervention_plan,
        "actioned":          r.actioned,
        "actioned_by":       r.actioned_by,
        "actioned_at":       r.actioned_at.isoformat() if r.actioned_at else None,
        "created_at":        r.created_at.isoformat() if r.created_at else None,
    }


@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    if "model" not in ml_components:
        raise HTTPException(status_code=503, detail="Model not loaded. Run trainer.py first.")

    contents = await file.read()
    df_raw = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")

    try:
        X = _preprocess_uploaded_df(df_raw)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {e}")

    try:
        t0            = time.perf_counter()
        threshold     = ml_components["threshold"]
        feature_names = ml_components["features"]
        X_arr         = X.values

        probabilities = ml_components["model"].predict_proba(X_arr)[:, 1]
        predictions   = (probabilities >= threshold).astype(int)

        shap_values = _get_explainer().shap_values(X_arr)
        top5_idx    = np.argsort(np.abs(shap_values), axis=1)[:, -5:][:, ::-1]

        rules        = db.query(InterventionRule).filter(InterventionRule.is_active == True).all()
        at_risk_mask = predictions.astype(bool)
        plans_series = pd.Series([""] * len(df_raw))

        if rules and at_risk_mask.any():
            at_risk_df = df_raw[at_risk_mask].copy()
            for rule in rules:
                if rule.feature not in at_risk_df.columns:
                    continue
                vals = pd.to_numeric(at_risk_df[rule.feature], errors="coerce")
                if rule.operator == ">":
                    mask = vals > rule.threshold
                elif rule.operator == "<":
                    mask = vals < rule.threshold
                else:
                    mask = vals == rule.threshold
                at_risk_df.loc[mask, "_plan"] = at_risk_df.loc[mask].get("_plan", "") + rule.plan_text + " | "
            if "_plan" in at_risk_df.columns:
                plans_series[at_risk_mask] = at_risk_df["_plan"].fillna("").str.rstrip(" | ")

        batch_id   = f"BATCH-{str(uuid.uuid4())[:6].upper()}"
        results    = []
        db_records = []

        for i in range(len(df_raw)):
            student_id   = f"STU-{i + 1000}"
            risk_prob    = round(float(probabilities[i]) * 100, 2)
            is_risk      = bool(predictions[i])
            plan         = plans_series.iloc[i] or ("Standard monitoring." if not is_risk else "")
            risk_factors = [
                {"feature": feature_names[j], "impact": round(float(shap_values[i, j]), 4)}
                for j in top5_idx[i]
            ]
            results.append({
                "student_id":        student_id,
                "risk_probability":  risk_prob,
                "is_at_risk":        is_risk,
                "intervention_plan": plan,
                "risk_factors":      risk_factors,
                "actioned":          False,
            })
            db_records.append({
                "batch_id":          batch_id,
                "student_id":        student_id,
                "risk_probability":  risk_prob,
                "is_at_risk":        is_risk,
                "intervention_plan": plan,
                "actioned":          False,
            })

        db.bulk_insert_mappings(PredictionRecord, db_records)
        db.commit()
        elapsed = round(time.perf_counter() - t0, 3)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    return {
        "message":        "Analysis complete and saved to database.",
        "batch_id":       batch_id,
        "uploaded_by":    current_user.username,
        "total_students": len(results),
        "at_risk_count":  sum(r["is_at_risk"] for r in results),
        "threshold_used": round(threshold, 4),
        "processing_ms":  int(elapsed * 1000),
        "data":           results,
    }


@app.get("/predict/history")
async def prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    records = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).all()
    return [_record_to_dict(r) for r in records]


@app.post("/predict/action/{student_id}")
async def mark_actioned(
    student_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    record = db.query(PredictionRecord).filter(
        PredictionRecord.student_id == student_id
    ).order_by(PredictionRecord.id.desc()).first()

    if not record:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")

    record.actioned    = True
    record.actioned_by = current_user.username
    record.actioned_at = datetime.now(timezone.utc)
    db.commit()

    return _record_to_dict(record)


@app.post("/predict/unaction/{student_id}")
async def mark_unactioned(
    student_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    record = db.query(PredictionRecord).filter(
        PredictionRecord.student_id == student_id
    ).order_by(PredictionRecord.id.desc()).first()

    if not record:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found.")

    record.actioned    = False
    record.actioned_by = None
    record.actioned_at = None
    db.commit()

    return _record_to_dict(record)


@app.post("/agent/draft-intervention")
async def draft_intervention(
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    student_data = {
        "student_id":   payload.get("student_id"),
        "risk_prob":    payload.get("risk_prob"),
        "risk_factors": payload.get("risk_factors", []),
    }
    plan = generate_intervention(student_data, db)
    return {"plan": plan}