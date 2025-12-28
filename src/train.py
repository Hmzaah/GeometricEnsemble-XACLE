import numpy as np
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import joblib
import os
from src.config import (
    WEIGHT_XGBOOST, WEIGHT_SVR, 
    XGB_PARAMS, SVR_PARAMS, MODEL_DIR
)

def train_ensemble_model(features, targets):
    """
    The 'Split-Brain' Training Pipeline.
    Stream A: XGBoost (Tree-based)
    Stream B: SVR (Manifold-based)
    """
    print(f"Loaded Unified Vector Shape: {features.shape}")
    
    # 1. Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    # 2. STREAM A: XGBoost
    print(f"Training Stream A: XGBoost (w={WEIGHT_XGBOOST})...")
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    # 3. STREAM B: SVR
    print(f"Training Stream B: SVR (w={WEIGHT_SVR})...")
    svr_model = SVR(**SVR_PARAMS)
    svr_model.fit(X_train, y_train)
    
    # 4. Fusion & Evaluation
    pred_xgb = xgb_model.predict(X_val)
    pred_svr = svr_model.predict(X_val)
    
    # Weighted Heterogeneous Ensemble
    final_pred = (pred_xgb * WEIGHT_XGBOOST) + (pred_svr * WEIGHT_SVR)
    
    # Calculate Metric
    xgb_score = spearmanr(y_val, pred_xgb).correlation
    svr_score = spearmanr(y_val, pred_svr).correlation
    ensemble_score = spearmanr(y_val, final_pred).correlation
    
    print(f"\nResults (Validation):")
    print(f"   XGBoost SRCC: {xgb_score:.4f}")
    print(f"   SVR SRCC:     {svr_score:.4f}")
    print(f"   ----------------------------")
    print(f"   FINAL SCORE:  {ensemble_score:.4f} (Target: ~0.653)")
    print(f"   ----------------------------")
    
    # 5. Save Models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_stream_a.pkl"))
    joblib.dump(svr_model, os.path.join(MODEL_DIR, "svr_stream_b.pkl"))
    print(f"Models saved to {MODEL_DIR}/")

if __name__ == "__main__":
    # Test Run with dummy data
    print("Running in test mode...")
    # Simulate the 9220-dim vector
    dummy_X = np.random.rand(100, 9220)
    dummy_y = np.random.rand(100)
    train_ensemble_model(dummy_X, dummy_y)
