import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_loader import load_and_preprocess_data


def generate_shap_summary(data_path, model_path):
    meta_path = model_path.replace('.json', '_meta.pkl')
    meta = joblib.load(meta_path)

    _, X_test, _, y_test, _ = load_and_preprocess_data(data_path)

    model = joblib.load(model_path.replace('.json', '.pkl'))

    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    out_path = model_path.replace('failsafe_xgb.json', 'shap_summary.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved SHAP summary -> {out_path}")

   
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    bar_path = model_path.replace('failsafe_xgb.json', 'shap_bar.png')
    plt.savefig(bar_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved SHAP bar plot -> {bar_path}")

    return shap_values, X_test


def explain_student(data_path, model_path, student_index: int):
    meta_path = model_path.replace('.json', '_meta.pkl')
    meta = joblib.load(meta_path)
    threshold = meta['threshold']

    _, X_test, _, y_test, _ = load_and_preprocess_data(data_path)

    model = joblib.load(model_path.replace('.json', '.pkl'))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    student_row = X_test.iloc[[student_index]]
    risk_score = model.predict_proba(student_row)[0, 1]
    flagged = risk_score >= threshold

    sv = shap_values[student_index]
    factor_df = pd.DataFrame({
        'feature':   X_test.columns,
        'shap_value': sv,
        'abs_shap':  np.abs(sv)
    }).sort_values('abs_shap', ascending=False)

    top_factors = list(zip(factor_df['feature'], factor_df['shap_value']))

    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[student_index],
        student_row,
        matplotlib=False
    )
    force_path = model_path.replace('failsafe_xgb.json', f'student_{student_index}_force.html')
    shap.save_html(force_path, force_plot)
    print(f"Saved force plot -> {force_path}")

    return {
        'student_index':    student_index,
        'risk_score':       round(float(risk_score), 4),
        'flagged':          bool(flagged),
        'threshold_used':   round(threshold, 4),
        'top_risk_factors': top_factors[:5]
    }


if __name__ == "__main__":
    DATA_FILE  = "../../data/student_data.csv"
    MODEL_FILE = "../../models/failsafe_xgb.json"

    print("Generating global SHAP analysis...")
    generate_shap_summary(DATA_FILE, MODEL_FILE)

    print("\nGenerating per-student explanation (student index 0)...")
    result = explain_student(DATA_FILE, MODEL_FILE, student_index=0)

    print(f"\nStudent {result['student_index']}:")
    print(f"  Risk Score : {result['risk_score']:.4f}")
    print(f"  Flagged    : {result['flagged']} (threshold={result['threshold_used']})")
    print(f"  Top Factors:")
    for feat, val in result['top_risk_factors']:
        direction = "up risk" if val > 0 else "down risk"
        print(f"    {feat:30s}  SHAP={val:+.4f}  {direction}")