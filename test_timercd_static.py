"""
Test script for TimeRCD with Static Covariates (Lat/Lon).

This script evaluates the TimeRCD model trained with 3 input features:
- Sea Level (normalized time series)
- Latitude (static covariate, constant across time)  
- Longitude (static covariate, constant across time)

Evaluation metrics: Confusion Matrix and Matthews Correlation Coefficient (MCC).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from tqdm import tqdm

# Add Time-RCD to path (priority)
sys.path.insert(0, os.path.join(os.getcwd(), "Time-RCD"))

from timercd_utils import FloodDatasetStatic
from models.time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from models.time_rcd.time_rcd_config import TimeRCDConfig

# Configuration
DATA_FILE = "foundation_data.pkl"
GEO_FILE = "analysis_plots/station_thresholds_geo.csv"

# Static Covariate Checkpoint (select the epoch to evaluate)

BATCH_SIZE = 32


def test(curr_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Test Dataset with Static Covariates
    print("Loading test data with Static Covariates (Lat/Lon)...")
    test_dataset = FloodDatasetStatic(DATA_FILE, GEO_FILE, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize Model with num_features=3 (Sea Level, Lat, Lon)
    print("Initializing TimeRCD with num_features=3...")
    config = TimeRCDConfig()
    config.ts_config.num_features = 3  # KEY: Must match training config
    config.ts_config.d_model = 512
    config.ts_config.patch_size = 16
    
    model = TimeSeriesPretrainModel(config).to(device)
    
    CHECKPOINT_PATH = f"checkpoints/timercd_static_covariate/static_covariate_epoch_{curr_epoch}.pth"
    # Load Weights
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from epoch {curr_epoch}: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Handle prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=True)  # strict=True since we expect exact match
        print("Model weights loaded successfully.")
    else:
        print(f"ERROR: Checkpoint {CHECKPOINT_PATH} not found!")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\nStarting Inference...")
    print(f"Input shape: (B, 504, 3) - [Sea_Level, Lat, Lon]")
    print("-" * 50)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 1. Get the Full Ground Truth (Keep this for checking answers later)
            full_ground_truth = batch['time_series'].to(device)  # (B, 504, 3)
            mask = batch['mask'].to(device)                      # (B, 504)
            
            # 2. Create the INPUT tensor (Clone it so we don't destroy GT)
            input_seq = full_ground_truth.clone()
            
            # 3. CRITICAL FIX: Zero out the Future (Simulate Real Testing)
            # Assuming mask is 1 for Future and 0 for History. 
            # If mask is boolean, use it directly.
            # We zero out ALL 3 channels (Sea Level, Lat, Lon) or just Sea Level?
            # Usually, you keep Lat/Lon (static) but zero out Sea Level.
            
            # Option A: Zero out everything in future (simplest)
            # input_seq[mask.bool()] = 0.0 
            
            # Option B: Zero out ONLY Sea Level (Channel 0) in future
            # This allows the model to still see Lat/Lon for the future days
            future_mask_indices = mask.bool() # Shape (B, 504)
            input_seq[future_mask_indices, 0] = 0.0 

            # 4. Create Attention Mask
            # We still allow full attention, but the future tokens are now ZEROS.
            # The model must use History + Lat/Lon to "in-paint" the zeros.
            attention_mask = torch.ones((input_seq.size(0), input_seq.size(1)), dtype=torch.bool).to(device)
            
            # 5. Forward Pass (Pass the MASKED input, not the full GT)
            embeddings = model(input_seq, attention_mask)
            
            # 6. Reconstruction
            reconstructed = model.reconstruction_head(embeddings) 
            reconstructed = reconstructed.view(input_seq.size(0), input_seq.size(1), 3)
            
            # 7. Evaluation Logic (Compare Prediction vs Ground Truth)
            for i in range(input_seq.size(0)):
                future_mask = mask[i].bool()
                
                # Get Predictions (from the model output)
                future_preds = reconstructed[i, :, 0][future_mask]
                
                # Get Ground Truth (from the original full_ground_truth variable)
                future_gt = full_ground_truth[i, :, 0][future_mask]
                
                # Check for flooding
                flood_pred = (future_preds > 0).any().item()
                flood_label = (future_gt > 0).any().item()
                
                all_preds.append(int(flood_pred))
                all_labels.append(int(flood_label))
                
    # Evaluation
    print("\nComputing metrics...")
    cm = confusion_matrix(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              No Flood | Flood")
    print(f"Actual No Flood  {cm[0,0]:5d}  | {cm[0,1]:5d}")
    print(f"Actual Flood     {cm[1,0]:5d}  | {cm[1,1]:5d}")
    print(f"\nMCC: {mcc:.4f}")
    
    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {
        'confusion_matrix': cm,
        'mcc': mcc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    total_epochs = 5
    for i in range(1, total_epochs+1):
        test(i)
