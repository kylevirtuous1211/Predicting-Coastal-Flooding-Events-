"""
Training script for TimeRCD with Static Covariates (Lat/Lon).

This script trains the TimeRCD model with 3 input features:
- Sea Level (normalized time series)
- Latitude (static covariate, constant across time)
- Longitude (static covariate, constant across time)

The model uses pretrained weights and fine-tunes with the additional static features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm

# Add Time-RCD to path (priority)
sys.path.insert(0, os.path.join(os.getcwd(), "Time-RCD"))

from timercd_utils import FloodDatasetStatic
from models.time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from models.time_rcd.time_rcd_config import TimeRCDConfig

# Configuration
DATA_FILE = "foundation_data.pkl"
GEO_FILE = "analysis_plots/station_thresholds_geo.csv"
CHECKPOINT_DIR = "checkpoints/timercd_static_covariate"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset with Static Covariates (Lat/Lon)
    print("Loading data with Static Covariates (Lat/Lon)...")
    train_dataset = FloodDatasetStatic(DATA_FILE, GEO_FILE, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Initialize Model with num_features=3 (Sea Level, Lat, Lon)
    print("Initializing TimeRCD with num_features=3...")
    config = TimeRCDConfig()
    config.ts_config.num_features = 3  # KEY CHANGE: 1 -> 3
    config.ts_config.d_model = 512
    config.ts_config.patch_size = 16
    
    model = TimeSeriesPretrainModel(config).to(device)
    
    # Load Pretrained Weights (strict=False to handle new BinaryAttentionBias)
    pretrained_path = "Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Handle potential prefix issues (e.g. 'module.' if DDP was used)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        # Non-strict load because:
        # 1. Pretrained has num_features=1, so no BinaryAttentionBias
        # 2. Our model has num_features=3, so it creates BinaryAttentionBias
        # 3. BinaryAttentionBias will be randomly initialized and trained
        print("Loading weights with strict=False (to handle new BinaryAttentionBias)...")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        if missing:
            print(f"  Missing: {missing[:5]}...")  # Show first 5
        print(f"Unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: Pretrained checkpoint not found at {pretrained_path}")
        print("Training from scratch (not recommended).")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nStarting Training with Static Covariates...")
    print(f"Input shape: (B, 504, 3) - [Sea_Level, Lat, Lon]")
    print("-" * 50)
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            # Batch: {'time_series': (B, 504, 3), 'mask': (B, 504), ...}
            
            time_series = batch['time_series'].to(device)  # (B, 504, 3)
            mask = batch['mask'].to(device)                # (B, 504)
            
            # Create Attention Mask (all valid)
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            embeddings = model(time_series, attention_mask) 
            
            # Calculate Loss (Reconstruction of all 3 features)
            loss = model.masked_reconstruction_loss(embeddings, time_series, mask)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
        
        # Save Checkpoint per epoch
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"static_covariate_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")
        
    print("\nTraining Complete.")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
