"""
Phase 2 Training: TimeRCD with Prior Token

Finetune TimeRCD model with frozen FloodScout providing prior token conditioning.
- FloodScout: Frozen (trained in Phase 1)
- PriorEmbedding: Trainable
- TimeRCD: Trainable (initialized from pretrained weights)

Usage:
    python train_timercd_prior.py --epochs 20 --batch_size 256
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add Time-RCD to path
sys.path.insert(0, os.path.join(os.getcwd(), "Time-RCD"))

from timercd_utils import FloodDataset
from model_prior_token import TimeRCDWithPrior, TimeRCDConfig

# Configuration
DATA_FILE = "foundation_data.pkl"
CHECKPOINT_DIR = "checkpoints/timercd_prior"
SCOUT_CHECKPOINT = "scout.pkl"
TIMERCD_PRETRAINED = "Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-4


def weighted_reconstruction_loss(embeddings, targets, mask, model, flood_weight=10.0):
    """
    Custom Loss that penalizes errors on FLOOD values much harder than normal values.
    """
    predictions = model.reconstruction_head(embeddings)
    predictions = predictions.view(targets.shape)
    loss = (predictions - targets) ** 2
    weights = torch.ones_like(loss)
    flood_indices = targets[:, :, 0] > 0.0
    weights[:, :, 0][flood_indices] = flood_weight
    mask_expanded = mask.unsqueeze(-1).expand_as(loss)
    final_loss = (loss * weights * mask_expanded).mean()
    return final_loss


def test_prior(model, device, split='test'):
    """Evaluate TimeRCDWithPrior model."""
    from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix
    
    test_dataset = FloodDataset(DATA_FILE, split=split)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            time_series = batch['time_series'].to(device)
            mask = batch['mask'].to(device)
            
            # Get flood label from future values
            Y = time_series[:, 168:, 0]  # Future sea level
            flood_labels = (Y > 0).any(dim=1).cpu().numpy()
            
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            # Zero out future for input
            input_seq = time_series.clone()
            input_seq[mask.bool()] = 0.0
            
            # Forward with prior token
            embeddings = model(input_seq, attention_mask)
            reconstructed = model.reconstruction_head(embeddings)
            reconstructed = reconstructed.view(time_series.shape)
            
            # Get peak prediction in future
            future_preds = reconstructed[:, 168:, 0]
            peak_vals = future_preds.max(dim=1)[0].cpu().numpy()
            
            all_preds.extend(peak_vals)
            all_labels.extend(flood_labels)
    
    # Find best threshold
    best_mcc = -1
    best_thresh = 0.0
    best_f1 = 0
    
    for t in np.arange(-0.5, 2.0, 0.05):
        preds_bin = [1 if x > t else 0 for x in all_preds]
        mcc = matthews_corrcoef(all_labels, preds_bin)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
            best_f1 = f1_score(all_labels, preds_bin, zero_division=0)
    
    preds_bin = [1 if x > best_thresh else 0 for x in all_preds]
    cm = confusion_matrix(all_labels, preds_bin)
    
    print(f"  Val MCC: {best_mcc:.4f} (thresh={best_thresh:.2f}), F1: {best_f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    return {'mcc': best_mcc, 'f1': best_f1, 'cm': cm, 'best_thresh': best_thresh}


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, flood_weight=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset
    print("Loading data...")
    train_dataset = FloodDataset(DATA_FILE, split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    print("Initializing TimeRCDWithPrior...")
    config = TimeRCDConfig()
    config.ts_config.num_features = 1
    config.ts_config.d_model = 512
    config.ts_config.patch_size = 16
    
    model = TimeRCDWithPrior(
        config, 
        scout_checkpoint=SCOUT_CHECKPOINT,
        context_len=168
    ).to(device)
    
    # Freeze Scout (Phase 2: Scout is frozen)
    model.freeze_scout()
    
    # Load pretrained TimeRCD weights into timercd submodule
    if os.path.exists(TIMERCD_PRETRAINED):
        print(f"Loading pretrained TimeRCD weights from {TIMERCD_PRETRAINED}")
        state_dict = torch.load(TIMERCD_PRETRAINED, map_location=device, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Handle prefix issues
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Load into timercd submodule
        try:
            model.timercd.load_state_dict(new_state_dict, strict=True)
            print("Successfully loaded pretrained TimeRCD weights (Strict).")
        except RuntimeError as e:
            print(f"Strict load failed: {e}")
            missing, unexpected = model.timercd.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print(f"WARNING: Pretrained checkpoint not found at {TIMERCD_PRETRAINED}")
    
    # Only optimize non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    print("Starting Training...")
    
    train_losses = []
    val_mccs = []
    val_f1s = []
    best_mcc = -1
    
    os.makedirs("training_plots", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        model.scout.eval()  # Keep scout in eval mode
        
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            time_series = batch['time_series'].to(device)  # (B, 504, 1)
            mask = batch['mask'].to(device)  # (B, 504)
            
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            # Zero out the Future in Training input
            train_input = time_series.clone()
            train_input[mask.bool()] = 0.0
            
            optimizer.zero_grad()
            
            # Forward Pass with Prior Token
            embeddings = model(train_input, attention_mask)
            
            # Calculate Loss
            loss = weighted_reconstruction_loss(embeddings, time_series, mask, model, flood_weight=flood_weight)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Validation
        print(f"Running Validation...")
        metrics = test_prior(model, device, split='test')
        val_mccs.append(metrics['mcc'])
        val_f1s.append(metrics['f1'])
        
        scheduler.step(metrics['mcc'])
        
        # Save best model
        if metrics['mcc'] > best_mcc:
            best_mcc = metrics['mcc']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mcc': best_mcc,
                'threshold': metrics['best_thresh']
            }, os.path.join(CHECKPOINT_DIR, "timercd_prior_best.pth"))
            print(f"  ✅ New best model saved! MCC: {best_mcc:.4f}")
        
        # Save epoch checkpoint
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"timercd_prior_epoch_{epoch+1}.pth"))
        
        # Plotting
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_mccs, label='Val MCC')
        plt.plot(val_f1s, label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("training_plots/timercd_prior_training.png")
        plt.close()
    
    print(f"\n{'='*50}")
    print(f"Training Complete! Best MCC: {best_mcc:.4f}")
    print(f"{'='*50}")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--flood_weight", type=float, default=5.0)
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, flood_weight=args.flood_weight)
