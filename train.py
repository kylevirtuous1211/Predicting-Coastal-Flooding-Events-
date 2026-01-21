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
    
    # 2. Check Class Imbalance
    total_samples = y_train.size
    pos_samples = np.sum(y_train)
    neg_samples = total_samples - pos_samples
    imbalance_ratio = neg_samples / pos_samples
    
    print(f"Class Distribution: {pos_samples} Positives (Floods) vs {neg_samples} Negatives")
    print(f"Imbalance Ratio: 1:{imbalance_ratio:.2f}")
    
    # 3. Define Model with Weighted Loss
    print(f"Applying scale_pos_weight={imbalance_ratio:.2f} to handle imbalance...")
    
    xgb_estimator = xgb.XGBRegressor(
        n_estimators=200,          # Increased trees
        learning_rate=0.05,        # Lower LR for better convergence
        max_depth=6,
        objective='binary:logistic',
        scale_pos_weight=imbalance_ratio, # CRITICAL: Fix for low F1
        n_jobs=-1,
        random_state=42
    )
    
    model = MultiOutputRegressor(xgb_estimator)
    
    # 4. Train
    print("Training XGBoost Model (this may take a moment)...")
    model.fit(X_train, y_train)
    print("Training Complete.")
    
    # 4. Save
    save_path = os.path.join(MODEL_DIR, MODEL_NAME)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
    
    # 5. Quick Evaluation on Validation Set
    # 5. Evaluate and Tune Threshold
    print("\n--- Tuning Threshold ---")
    y_pred_prob = model.predict(X_val)
    y_true_flat = y_val.flatten()
    
    best_thresh = 0.5
    best_f1 = 0.0
    best_metrics = {}
    
    # Check thresholds from 0.1 to 0.95
    for thresh in np.arange(0.1, 0.96, 0.05):
        y_pred = (y_pred_prob > thresh).astype(int)
        y_pred_flat = y_pred.flatten()
        
        f1 = f1_score(y_true_flat, y_pred_flat)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                'acc': accuracy_score(y_true_flat, y_pred_flat),
                'f1': f1,
                'mcc': matthews_corrcoef(y_true_flat, y_pred_flat)
            }
            
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"Overall Accuracy: {best_metrics['acc']:.4f}")
    print(f"Overall F1 Score: {best_metrics['f1']:.4f}")
    print(f"Overall MCC:      {best_metrics['mcc']:.4f}")
    
    # Save the best threshold to a simple text file for model.py to read (or hardcode)
    with open(os.path.join(MODEL_DIR, "best_threshold.txt"), "w") as f:
        f.write(str(best_thresh))

if __name__ == "__main__":
    train()
