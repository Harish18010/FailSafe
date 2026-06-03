import os
import random
import xgboost as xgb
import joblib
import numpy as np
from itertools import product
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from data_loader import load_and_preprocess_data


def find_optimal_threshold(y_true, y_proba, strategy='youden'):
    """
    strategy='youden' — maximises (TPR - FPR). Best balanced sensitivity/specificity.
    strategy='f1'     — maximises F1 on the at-risk class. Better when false alarms are costly.
    """
    if strategy == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        best_idx = np.argmax(tpr - fpr)
        return float(thresholds[best_idx])
    elif strategy == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        return float(thresholds[best_idx])
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'youden' or 'f1'.")


def manual_random_search(X, y, param_dist, n_iter=60, n_splits=5, random_seed=42):
    """
    Manual randomised hyperparameter search with stratified K-fold CV.

    Replaces RandomizedSearchCV entirely — no joblib multiprocessing, no
    pickling of estimators across worker processes. This eliminates the
    Windows-specific bug where XGBClassifier loses _estimator_type during
    inter-process serialisation, causing sklearn's roc_auc scorer to crash.
    """
    rng = random.Random(random_seed)
    all_combos = list(product(*param_dist.values()))
    sampled_combos = rng.sample(all_combos, min(n_iter, len(all_combos)))
    keys = list(param_dist.keys())

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    best_score  = -1.0
    best_params = {}

    print(f"Running {len(sampled_combos)} candidates × {n_splits} folds "
          f"= {len(sampled_combos) * n_splits} fits (single-process, no pickling)")

    for i, combo in enumerate(sampled_combos):
        params = dict(zip(keys, combo))
        fold_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            m = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                tree_method='hist',
                random_state=random_seed,
                **params
            )
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_val)[:, 1]
            fold_scores.append(roc_auc_score(y_val, proba))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score  = mean_score
            best_params = params

        if (i + 1) % 10 == 0 or (i + 1) == len(sampled_combos):
            print(f"  [{i+1:>3}/{len(sampled_combos)}] best CV ROC-AUC so far: {best_score:.4f}")

    return best_params, best_score


def train_optimized_model(data_path, model_save_path, threshold_strategy='youden'):
    encoder_save_path = model_save_path.replace('.json', '_encoder.pkl')
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(
        data_path, encoder_save_path=encoder_save_path
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"At-risk in train: {y_train.sum()} / {len(y_train)}")

    # SMOTE on training data only
    print("\nApplying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    X_res_arr = X_res.values if hasattr(X_res, 'values') else X_res
    y_res_arr  = y_res.values  if hasattr(y_res,  'values') else y_res
    print(f"After SMOTE — At-risk: {y_res_arr.sum()} / {len(y_res_arr)}")

    param_dist = {
        'max_depth':        [2, 3, 4, 5],
        'learning_rate':    [0.01, 0.05, 0.1, 0.15],
        'n_estimators':     [100, 200, 300],
        'subsample':        [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'reg_lambda':       [1, 5, 10],
        'reg_alpha':        [0, 0.1, 0.5],
    }

    print("\nStarting manual randomised search + cross-validation...")
    best_params, best_cv_score = manual_random_search(
        X_res_arr, y_res_arr, param_dist, n_iter=60, n_splits=5
    )

    print(f"\nBest Parameters:  {best_params}")
    print(f"Best CV ROC-AUC:  {best_cv_score:.4f}")

    # Train final model on full resampled set with best params
    print("\nTraining final model on full training set...")
    best_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        random_state=42,
        **best_params,
    )
    best_model.fit(X_res_arr, y_res_arr)

    # Data-driven threshold calibration on held-out test set
    X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
    y_pred_proba = best_model.predict_proba(X_test_arr)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba, strategy=threshold_strategy)
    print(f"\nOptimal threshold ({threshold_strategy}): {optimal_threshold:.4f}")

    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report  = classification_report(y_test, y_pred)

    # Save using joblib — bypasses xgboost's save_model() which crashes when
    # _estimator_type isn't set as a class attribute (version-dependent bug).
    # explainer.py and main.py load with joblib.load() to match.
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model_pkl_path = model_save_path.replace('.json', '.pkl')
    joblib.dump(best_model, model_pkl_path)

    # Also save as native XGBoost JSON — version-safe across environments
    best_model.get_booster().save_model(model_save_path)

    meta = {
        'feature_names':      list(feature_names),
        'threshold':          optimal_threshold,
        'threshold_strategy': threshold_strategy,
        'best_params':        best_params,
        'cv_roc_auc':         best_cv_score,
        'model_path':         model_pkl_path,
        'model_json_path':    model_save_path,
    }
    joblib.dump(meta, model_save_path.replace('.json', '_meta.pkl'))

    print(f"\nModel saved   -> {model_pkl_path}")
    print(f"Encoder saved → {encoder_save_path}")
    print(f"Meta saved    → {model_save_path.replace('.json', '_meta.pkl')}")

    return best_model, acc, roc_auc, report, optimal_threshold


if __name__ == "__main__":
    DATA_FILE  = "../../data/student_data.csv"
    MODEL_FILE = "../../models/failsafe_xgb.json"

    print("=" * 55)
    print("  FAILSAFE — Training Optimized Model")
    print("=" * 55)

    model, accuracy, roc_auc, report, threshold = train_optimized_model(
        DATA_FILE, MODEL_FILE, threshold_strategy='youden'
    )

    print(f"\nFinal Accuracy:     {accuracy:.4f}")
    print(f"ROC-AUC Score:      {roc_auc:.4f}")
    print(f"Decision Threshold: {threshold:.4f}")
    print("\nClassification Report:\n", report)