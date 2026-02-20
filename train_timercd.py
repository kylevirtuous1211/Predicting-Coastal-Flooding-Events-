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
DATA_FILE = "foundation_data_deep_105d.pkl"
CHECKPOINT_DIR = "checkpoints/timercd_finetune/75days"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 64 # Reduced from 256 for longer context
EPOCHS = 60     # Increased from 20
LEARNING_RATE = 1e-4
CONTEXT_LEN = 1800
PATCH_SIZE = 21
FLOOD_WEIGHT = 8.0

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

def train(resume_checkpoint=None, epochs=EPOCHS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset
    print("Loading data...")
    # Enable Augmentation for Training
    train_dataset = FloodDataset(DATA_FILE, split='train', context_len=CONTEXT_LEN, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    print("Initializing TimeRCD...")
    config = TimeRCDConfig()
    config.ts_config.num_features = 1
    config.ts_config.d_model = 512 # Default
    config.ts_config.patch_size = PATCH_SIZE 
    
    model = TimeSeriesPretrainModel(config).to(device)
    
    start_epoch = 0
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        state_dict = torch.load(resume_checkpoint, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Handle prefix issues
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Try to load checkpoint
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("Checkpoint loaded (Strict).")
        except RuntimeError as e:
            print(f"Strict load failed: {e}")
            print("Attempting non-strict load with shape mismatch filtering...")
            
            # Filter out keys with shape mismatch
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for k, v in new_state_dict.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_state_dict[k].shape}")
                else:
                    print(f"Skipping {k} as it is not in the model.")
            
            missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded with filtering. Missing: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")
        
        # Try to infer start epoch from filename (e.g., timercd_epoch_40.pth)
        try:
            basename = os.path.basename(resume_checkpoint)
            parts = basename.replace('.pth', '').split('_')
            if 'epoch' in parts:
                idx = parts.index('epoch')
                if idx + 1 < len(parts):
                    start_epoch = int(parts[idx+1])
                    print(f"Resuming at Epoch {start_epoch + 1}")
        except:
            print("Could not infer start epoch from filename.")
            
    else:
        # Load Pretrained Weights (Original Logic)
        pretrained_path = "Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            try:
                model.load_state_dict(new_state_dict, strict=True)
                print("Successfully loaded pretrained weights (Strict).")
            except RuntimeError as e:
                print(f"Strict load failed: {e}")
                # Filter out keys with shape mismatch
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                for k, v in new_state_dict.items():
                    if k in model_state_dict:
                        if v.shape == model_state_dict[k].shape:
                            filtered_state_dict[k] = v
                missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Loaded with filtering. Missing: {len(missing)}")
        else:
            print(f"WARNING: Pretrained checkpoint not found at {pretrained_path}. Training from scratch.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Training...")
    model.train()
    
    train_losses = []
    val_mccs = []
    val_f1s = []
    
    os.makedirs("training_plots", exist_ok=True)
    
    # Adjust total epochs relative to start
    total_epochs = start_epoch + epochs
    
    for epoch in range(start_epoch, total_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")):
            time_series = batch['time_series'].to(device) # (B, 504, 1)
            mask = batch['mask'].to(device)               # (B, 504)
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            train_input = time_series.clone()
            train_input[mask.bool()] = 0.0
            
            optimizer.zero_grad()
            embeddings = model(train_input, attention_mask) 
            loss = weighted_reconstruction_loss(embeddings, time_series, mask, model, flood_weight=FLOOD_WEIGHT)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{total_epochs}, Loss: {avg_loss:.6f}")
        train_losses.append(avg_loss)
        
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"timercd_epoch_{epoch+1}.pth"))
        
        print(f"Running Validation for Epoch {epoch+1}...")
        metrics = test(model=model, device=device, split='test', 
                       context_len=CONTEXT_LEN, patch_size=PATCH_SIZE, data_file=DATA_FILE)
        val_mccs.append(metrics['mcc'])
        val_f1s.append(metrics['f1'])
        model.train()
        
        # Plotting code follows (omitted for brevity, can remain as is if not in replacement chunk)
        # Side-by-side Plotting to match TimeRCD_prior
        plt.figure(figsize=(12, 4))
        
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        title_str = f"TimeRCD (P={PATCH_SIZE}, W={FLOOD_WEIGHT}, C={CONTEXT_LEN}h)\n{date_str}"
        
        plt.subplot(1, 2, 1)
        # Handle resume case for x-axis
        x_range_train = range(start_epoch + 1, epoch + 2)
        # Note: train_losses is appended every epoch, but if we resume, we only have new losses.
        # Plotting against correct epoch numbers.
        plt.plot(x_range_train, train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss\n{title_str}', fontsize=10)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        x_range_val = range(start_epoch + 1, epoch + 2)
        plt.plot(x_range_val, val_mccs, label='Val MCC')
        plt.plot(x_range_val, val_f1s, label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'Validation Metrics\n{title_str}', fontsize=10)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("training_plots/timercd_finetune_training.png")
        plt.close()
        
    print("Training Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of ADDITIONAL epochs to train")
    args = parser.parse_args()
    
    train(resume_checkpoint=args.resume, epochs=args.epochs)
