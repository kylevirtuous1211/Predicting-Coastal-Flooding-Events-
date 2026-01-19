import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np
import os
import joblib
from dataset import FloodDataset

# Configuration
MODEL_DIR = "checkpoints"
MODEL_NAME = "xgb_global_model.pkl"

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load Data
    ds = FloodDataset()
    X_train, y_train = ds.get_train_data()
    X_val, y_val = ds.get_val_data()
    
    print(f"Training on {len(X_train)} samples, Validating on {len(X_val)} samples.")
    
    # 2. Define Model
    # We use MultiOutputRegressor to fit one regressor per target day (14 outputs)
    xgb_estimator = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='binary:logistic', 
        n_jobs=-1,
        random_state=42
    )
    
    model = MultiOutputRegressor(xgb_estimator)
    
    # 3. Train
    print("Training XGBoost Model (this may take a moment)...")
    model.fit(X_train, y_train)
    print("Training Complete.")
    
    # 4. Save
    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
    
    # 5. Quick Evaluation on Validation Set
    print("\n--- Validation Scores ---")
    y_pred_prob = model.predict(X_val)
    # Convert probabilities to binary class (Threshold 0.5)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Flatten for global metrics
    y_true_flat = y_val.flatten()
    y_pred_flat = y_pred.flatten()
    
    acc = accuracy_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)
    mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
    
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Overall F1 Score: {f1:.4f}")
    print(f"Overall MCC:      {mcc:.4f}")

if __name__ == "__main__":
    train()
