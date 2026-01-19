import joblib
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from dataset import FloodDataset

# Configuration
MODEL_PATH = os.path.join("checkpoints", "xgb_global_model.pkl")

def inference():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run train.py first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    print("Loading test/validation data...")
    ds = FloodDataset()
    X_test, y_test = ds.get_val_data() # Using validation set as test here
    
    print("Running Inference...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Detailed Analysis
    print("\n--- Detailed Evaluation ---")
    
    # Global Metrics
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_flat, y_pred_flat))
    
    print("\nClassification Report:")
    print(classification_report(y_test_flat, y_pred_flat))
    
    # Day-by-Day Performance
    print("\n--- F1 Score per Forecast Day (Day 1 to 14) ---")
    f1_per_day = []
    from sklearn.metrics import f1_score
    
    for d in range(14):
        f1 = f1_score(y_test[:, d], y_pred[:, d])
        f1_per_day.append(f1)
        print(f"Day {d+1:02d}: {f1:.4f}")
        
    avg_f1 = np.mean(f1_per_day)
    print(f"\nAverage Daily F1: {avg_f1:.4f}")
    
    # Save predictions if needed
    # np.save("val_predictions.npy", y_pred)

if __name__ == "__main__":
    inference()
