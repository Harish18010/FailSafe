import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import os
import joblib

def load_and_preprocess_data(filepath, target_threshold=10, encoder_save_path=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, sep=None, engine='python')

    # --- Feature Engineering: Grade Trajectory ---
    # G1 and G2 are mid-term grades — strong predictors.
    # We don't use them raw (to avoid leaking G3), but we extract
    # trend and average signals before dropping them.
    if 'G1' in df.columns and 'G2' in df.columns:
        df['grade_trend'] = df['G2'] - df['G1']       # positive = improving, negative = declining
        df['mid_avg'] = (df['G1'] + df['G2']) / 2     # baseline performance level

    # Target: at-risk if final grade G3 < threshold
    df['at_risk'] = (df['G3'] < target_threshold).astype(int)

    # Drop source grade columns and target
    drop_cols = ['G1', 'G2', 'G3', 'at_risk']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['at_risk']

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Train/test split BEFORE encoding to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit OrdinalEncoder only on training data
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = enc.transform(X_test[cat_cols])

        # Save encoder so the FastAPI endpoint can use it at inference time
        if encoder_save_path:
            os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)
            joblib.dump({'encoder': enc, 'cat_cols': cat_cols}, encoder_save_path)

    return X_train, X_test, y_train, y_test, X.columns