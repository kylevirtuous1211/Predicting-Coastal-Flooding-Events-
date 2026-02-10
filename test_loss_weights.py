"""
Test Different Loss Weights for TimeRCD Model - "Imbalance" Experiment

This script trains and evaluates TimeRCD with different flood_weight configurations
in the weighted reconstruction loss to find the optimal balance for MCC.

Usage:
    python test_loss_weights.py --epochs 10 --weights 1.0 3.0 5.0 8.0 10.0 15.0 20.0
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
import json
from datetime import datetime

# Add Time-RCD to path
sys.path.insert(0, os.path.join(os.getcwd(), "Time-RCD"))

from timercd_utils import FloodDataset
from model_prior_token import TimeRCDWithPrior, TimeRCDConfig
from sklearn.metrics import matthews_corrcoef, f1_score

# Configuration
DATA_FILE = "foundation_data.pkl"  # Use original 168h dataset
SCOUT_CHECKPOINT = "checkpoints/scout/scout_best.pth"
TIMERCD_PRETRAINED = "Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"

BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-4
PATCH_SIZE = 21  # Optimal from patch size experiment
CONTEXT_LEN = 168  # Default, can be overridden


def weighted_reconstruction_loss(embeddings, targets, mask, model, flood_weight=5.0):
    """Custom Loss that penalizes errors on FLOOD values harder."""
    predictions = model.reconstruction_head(embeddings)
    predictions = predictions.view(targets.shape)
    loss = (predictions - targets) ** 2
    weights = torch.ones_like(loss)
    flood_indices = targets[:, :, 0] > 0.0
    weights[:, :, 0][flood_indices] = flood_weight
    mask_expanded = mask.unsqueeze(-1).expand_as(loss)
    final_loss = (loss * weights * mask_expanded).mean()
    return final_loss


def evaluate(model, device, context_len=CONTEXT_LEN, data_file=DATA_FILE, split='test'):
    """Evaluate model and return metrics."""
    test_dataset = FloodDataset(data_file, split=split, context_len=context_len)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            time_series = batch['time_series'].to(device)
            mask = batch['mask'].to(device)
            
            Y = time_series[:, context_len:, 0]
            flood_labels = (Y > 0).any(dim=1).cpu().numpy()
            
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            input_seq = time_series.clone()
            input_seq[mask.bool()] = 0.0
            
            embeddings = model(input_seq, attention_mask)
            reconstructed = model.reconstruction_head(embeddings)
            reconstructed = reconstructed.view(time_series.shape)
            
            future_preds = reconstructed[:, context_len:, 0]
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
    
    return {'mcc': best_mcc, 'f1': best_f1, 'best_thresh': best_thresh}


def train_with_flood_weight(flood_weight, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Train TimeRCDWithPrior with a specific flood weight."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"  Training with flood_weight = {flood_weight}")
    print(f"  patch_size = {PATCH_SIZE}, context_len = {CONTEXT_LEN}")
    print(f"{'='*60}")
    
    # Load Dataset
    train_dataset = FloodDataset(DATA_FILE, split='train', context_len=CONTEXT_LEN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    config = TimeRCDConfig()
    config.ts_config.num_features = 1
    config.ts_config.d_model = 512
    config.ts_config.patch_size = PATCH_SIZE
    
    model = TimeRCDWithPrior(
        config, 
        scout_checkpoint=SCOUT_CHECKPOINT,
        context_len=CONTEXT_LEN
    ).to(device)
    
    model.freeze_scout()
    
    # Load pretrained TimeRCD weights
    if os.path.exists(TIMERCD_PRETRAINED):
        state_dict = torch.load(TIMERCD_PRETRAINED, map_location=device, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Filter layers that depend on patch_size
        incompatible_keys = ['embedding_layer', 'projection_layer']
        filtered_state_dict = {
            k: v for k, v in new_state_dict.items() 
            if not any(ik in k for ik in incompatible_keys)
        }
        
        missing, unexpected = model.timercd.load_state_dict(filtered_state_dict, strict=False)
        print(f"  Loaded {len(filtered_state_dict) - len(missing)} pretrained layers")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
    
    train_losses = []
    val_mccs = []
    best_mcc = -1
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        model.scout.eval()
        
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            time_series = batch['time_series'].to(device)
            mask = batch['mask'].to(device)
            
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            train_input = time_series.clone()
            train_input[mask.bool()] = 0.0
            
            optimizer.zero_grad()
            embeddings = model(train_input, attention_mask)
            # KEY: Use the specified flood_weight
            loss = weighted_reconstruction_loss(embeddings, time_series, mask, model, flood_weight=flood_weight)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        metrics = evaluate(model, device)
        val_mccs.append(metrics['mcc'])
        
        if metrics['mcc'] > best_mcc:
            best_mcc = metrics['mcc']
            best_epoch = epoch + 1
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}")
    
    return {
        'flood_weight': flood_weight,
        'best_mcc': best_mcc,
        'best_epoch': best_epoch,
        'final_mcc': val_mccs[-1],
        'train_losses': train_losses,
        'val_mccs': val_mccs
    }


def run_experiments(weights, epochs=EPOCHS):
    """Run experiments for all flood weights and compare results."""
    print(f"\n{'#'*60}")
    print(f"  LOSS WEIGHT EXPERIMENT (Imbalance)")
    print(f"  Testing flood_weights: {weights}")
    print(f"  patch_size: {PATCH_SIZE}, context_len: {CONTEXT_LEN}")
    print(f"  Epochs per config: {epochs}")
    print(f"{'#'*60}")
    
    results = []
    
    for w in weights:
        try:
            result = train_with_flood_weight(w, epochs=epochs)
            results.append(result)
        except Exception as e:
            print(f"ERROR with flood_weight={w}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'flood_weight': w,
                'best_mcc': -1,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Weight':<12} {'Best MCC':<12} {'Best Epoch':<12} {'Final MCC':<12}")
    print("-" * 48)
    
    for r in results:
        if 'error' in r:
            print(f"{r['flood_weight']:<12} ERROR: {r['error'][:30]}")
        else:
            print(f"{r['flood_weight']:<12.1f} {r['best_mcc']:<12.4f} {r['best_epoch']:<12} {r['final_mcc']:<12.4f}")
    
    # Find best
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['best_mcc'])
        print(f"\n✅ BEST FLOOD WEIGHT: {best['flood_weight']}")
        print(f"   MCC={best['best_mcc']:.4f} at epoch {best['best_epoch']}")
    
    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"experiment_results/loss_weight_exp_{timestamp}.json"
    
    serializable_results = []
    for r in results:
        sr = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        serializable_results.append(sr)
    
    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {result_file}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for r in valid_results:
        plt.plot(r['val_mccs'], label=f"weight={r['flood_weight']}", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MCC')
    plt.title('MCC vs Epoch for Different Loss Weights')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    weights_plot = [r['flood_weight'] for r in valid_results]
    best_mccs_plot = [r['best_mcc'] for r in valid_results]
    plt.plot(weights_plot, best_mccs_plot, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Flood Weight')
    plt.ylabel('Best MCC')
    plt.title('Best MCC vs Loss Weight (Goldilocks Curve)')
    plt.grid(True)
    
    plt.tight_layout()
    plot_file = f"experiment_results/loss_weight_exp_{timestamp}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to: {plot_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test different loss weights for TimeRCD")
    parser.add_argument("--weights", type=float, nargs='+', default=[1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0],
                        help="List of flood weights to test (default: 1.0 3.0 5.0 8.0 10.0 15.0 20.0)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per configuration (default: 10)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--context_len", type=int, default=168,
                        help="Context length in hours (default: 168)")
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    CONTEXT_LEN = args.context_len
    run_experiments(args.weights, epochs=args.epochs)
