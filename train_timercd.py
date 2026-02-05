import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt

# Add Time-RCD to path (priority)
sys.path.insert(0, os.path.join(os.getcwd(), "Time-RCD"))

from test_timercd import test

from timercd_utils import FloodDataset
from models.time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from models.time_rcd.time_rcd_config import TimeRCDConfig

# Configuration
DATA_FILE = "foundation_data.pkl"
CHECKPOINT_DIR = "checkpoints/timercd_finetune"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-4

def weighted_reconstruction_loss(embeddings, targets, mask, model, flood_weight=10.0):
    """
    Custom Loss that penalizes errors on FLOOD values much harder than normal values.
    """
    # 1. Get the model's prediction (Reconstruction)
    # The model outputs embeddings, we need to project them back to values
    predictions = model.reconstruction_head(embeddings) 
    # 2. Reshape to match targets (B, 504, 3, 1) -> (B, 504, 3)
    predictions = predictions.view(targets.shape)
    # 3. Calculate Squared Error (MSE)
    # targets is the Ground Truth (time_series)
    loss = (predictions - targets) ** 2
    # 4. Create the Weight Map
    # Default weight = 1.0
    weights = torch.ones_like(loss)
    flood_indices = targets[:, :, 0] > 0.0 
    # Apply the penalty weight to the Sea Level channel (Channel 0) at those times
    weights[:, :, 0][flood_indices] = flood_weight
    # 5. Apply the Mask (Only train on the Future/Masked part)
    # mask is 1 for future, 0 for history
    # We multiply by mask to ignore history errors
    mask_expanded = mask.unsqueeze(-1).expand_as(loss) # (B, 504, 3)
    final_loss = (loss * weights * mask_expanded).mean()

    return final_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset
    print("Loading data...")
    train_dataset = FloodDataset(DATA_FILE, split='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model (Load Pretrained if available logic here, for now strictly init)
    # TODO: Load actual pretrained weights if user provides path or if they exist in standard loc
    # For now, we initialize from scratch as placeholder for "Pretrained" structure availability
    print("Initializing TimeRCD...")
    config = TimeRCDConfig()
    config.ts_config.num_features = 1
    config.ts_config.d_model = 512 # Default
    # Ensure seq_len covers our 504 length
    # TimeRCD usually handles variable length or has max_seq_len
    
    # Initialize Model
    print("Initializing TimeRCD...")
    config = TimeRCDConfig()
    config.ts_config.num_features = 1
    config.ts_config.d_model = 512 # Default
    config.ts_config.patch_size = 16 # As seen in model_wrapper.py for Univariate
    
    model = TimeSeriesPretrainModel(config).to(device)
    
    # Load Pretrained Weights
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
                
        # TimeRCD pretrained model has specific keys. 
        # We need to match our model structure.
        # If strict loading fails, we might need to filter keys or allow non-strict.
        # Attempting strict load first, then loose.
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("Successfully loaded pretrained weights (Strict).")
        except RuntimeError as e:
            print(f"Strict load failed: {e}")
            print("Attempting non-strict load...")
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: Pretrained checkpoint not found at {pretrained_path}. Training from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Training...")
    model.train()
    
    train_losses = []
    val_mccs = []
    val_f1s = []
    
    os.makedirs("training_plots", exist_ok=True)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            # Batch: {'time_series': (B, 504, 1), 'mask': (B, 504), 'threshold': ...}
            
            time_series = batch['time_series'].to(device) # (B, 504, 1)
            mask = batch['mask'].to(device)               # (B, 504) -> 1 where we want to predict (Future)
            
            # Create Attention Mask (assuming all valid since fixed length)
            # Shape: (B, 504)
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            # CRITICAL FIX: Zero out the Future in Training too!
            # Otherwise we are training an Autoencoder (Copy task), not a Forecaster.
            train_input = time_series.clone()
            train_input[mask.bool()] = 0.0
            
            optimizer.zero_grad()
            
            # Forward Pass
            # TimeSeriesPretrainModel (from TimeRCD_pretrain_multi.py) forward returns embeddings
            # We must pass attention_mask as the second argument
            embeddings = model(train_input, attention_mask) 
            
            # Calculate Loss
            # We want to reconstruct the MASKED part (The Future)
            loss = weighted_reconstruction_loss(embeddings, time_series, mask, model, flood_weight=5.0)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
        
        train_losses.append(avg_loss)
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"timercd_epoch_{epoch+1}.pth"))
        
        # Validation
        print(f"Running Validation for Epoch {epoch+1}...")
        metrics = test(model=model, device=device, split='test')
        val_mccs.append(metrics['mcc'])
        val_f1s.append(metrics['f1'])
        
        model.train() # Switch back to train mode
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+2), train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig("training_plots/loss.png")
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+2), val_mccs, label='MCC')
        plt.plot(range(1, epoch+2), val_f1s, label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig("training_plots/metrics.png")
        plt.close()
        
    print("Training Complete.")

if __name__ == "__main__":
    train()
