"""
Phase 1 Training: FloodScout

Train the FloodScout module separately with supervised learning.
- Input: 7-day history (168 hours)
- Target: Binary label (did future flood?)
- Loss: Focal Loss (handles class imbalance)

Usage:
    python train_scout.py --epochs 20 --batch_size 256
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from model_prior_token import FloodScout, FloodDataset

# Configuration
DATA_FILE = "foundation_data.pkl"
CHECKPOINT_DIR = "checkpoints/scout"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive class
        self.gamma = gamma  # Focusing parameter
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, 1) predictions from sigmoid
            targets: (B,) binary labels
        """
        inputs = inputs.view(-1)
        targets = targets.float()
        
        p = inputs
        ce_loss = -targets * torch.log(p + 1e-8) - (1 - targets) * torch.log(1 - p + 1e-8)
        
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


def train(epochs: int = 20, batch_size: int = 256, lr: float = 1e-3, patience: int = 5):
    """Train FloodScout with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Dataset
    print("Loading data...")
    train_dataset = FloodDataset(DATA_FILE, split='train')
    test_dataset = FloodDataset(DATA_FILE, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Check class distribution
    train_labels = [train_dataset[i]['flood_label'] for i in range(len(train_dataset))]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    print(f"Training set: {len(train_labels)} samples")
    print(f"  Positive (flood): {pos_count} ({100*pos_count/len(train_labels):.1f}%)")
    print(f"  Negative (no flood): {neg_count} ({100*neg_count/len(train_labels):.1f}%)")
    
    # Initialize model
    print("Initializing FloodScout...")
    model = FloodScout(input_len=168).to(device)
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Weight positive class more
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training tracking
    train_losses = []
    val_mccs = []
    val_aucs = []
    best_mcc = -1
    patience_counter = 0
    
    os.makedirs("training_plots", exist_ok=True)
    
    print("Starting Training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_preds = []
        train_labels_batch = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            time_series = batch['time_series'].to(device)  # (B, 504, 1)
            flood_labels = batch['flood_label'].to(device)  # (B,)
            
            # Extract ONLY history for scout
            history = time_series[:, :168, :]  # (B, 168, 1)
            
            optimizer.zero_grad()
            
            # Forward pass
            probs = model(history)  # (B, 1)
            
            # Loss
            loss = criterion(probs, flood_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_preds.extend(probs.detach().cpu().numpy().flatten())
            train_labels_batch.extend(flood_labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Compute training metrics
        train_preds_bin = [1 if p > 0.5 else 0 for p in train_preds]
        train_mcc = matthews_corrcoef(train_labels_batch, train_preds_bin)
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                time_series = batch['time_series'].to(device)
                flood_labels = batch['flood_label']
                
                history = time_series[:, :168, :]
                probs = model(history)
                
                val_preds.extend(probs.cpu().numpy().flatten())
                val_labels_list.extend(flood_labels.numpy())
        
        # Find optimal threshold for MCC
        best_t_mcc = -1
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            preds_bin = [1 if p > t else 0 for p in val_preds]
            mcc = matthews_corrcoef(val_labels_list, preds_bin)
            if mcc > best_t_mcc:
                best_t_mcc = mcc
                best_t = t
        
        val_mcc = best_t_mcc
        val_mccs.append(val_mcc)
        
        try:
            val_auc = roc_auc_score(val_labels_list, val_preds)
        except:
            val_auc = 0.5
        val_aucs.append(val_auc)
        
        # Update scheduler
        scheduler.step(val_mcc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_loss:.4f}, Train MCC: {train_mcc:.4f}")
        print(f"  Val MCC: {val_mcc:.4f} (thresh={best_t:.2f}), Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mcc': val_mcc,
                'threshold': best_t
            }, os.path.join(CHECKPOINT_DIR, "scout_best.pth"))
            print(f"  ✅ New best model saved! MCC: {val_mcc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint each epoch
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"scout_epoch_{epoch+1}.pth"))
        
        # Plot progress
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Focal Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_mccs, label='Val MCC')
        plt.plot(val_aucs, label='Val AUC')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("training_plots/scout_training.png")
        plt.close()
    
    # Final evaluation with best model
    print("\n" + "="*50)
    print("Final Evaluation with Best Model")
    print("="*50)
    
    best_state = torch.load(os.path.join(CHECKPOINT_DIR, "scout_best.pth"))
    model.load_state_dict(best_state['model_state_dict'])
    best_thresh = best_state['threshold']
    
    model.eval()
    final_preds = []
    final_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            time_series = batch['time_series'].to(device)
            flood_labels = batch['flood_label']
            
            history = time_series[:, :168, :]
            probs = model(history)
            
            final_preds.extend(probs.cpu().numpy().flatten())
            final_labels.extend(flood_labels.numpy())
    
    final_preds_bin = [1 if p > best_thresh else 0 for p in final_preds]
    final_mcc = matthews_corrcoef(final_labels, final_preds_bin)
    final_f1 = f1_score(final_labels, final_preds_bin)
    cm = confusion_matrix(final_labels, final_preds_bin)
    
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"Final MCC: {final_mcc:.4f}")
    print(f"Final F1: {final_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Copy best model to standard location
    import shutil
    shutil.copy(
        os.path.join(CHECKPOINT_DIR, "scout_best.pth"),
        "scout.pkl"
    )
    print(f"\n✅ Best model copied to scout.pkl")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
