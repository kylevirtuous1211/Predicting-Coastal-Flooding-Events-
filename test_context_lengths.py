"""
Test Different Context Lengths for TimeRCD Model - "Tidal Physics" Experiment

This script trains and evaluates TimeRCD with different context_len configurations
to find the optimal look-back window for capturing tidal cycles.

Previous results (7-30 day windows):
    168h (7d):  MCC=0.2735 @ epoch 15
    336h (14d): MCC=0.3112 @ epoch 19
    720h (30d): MCC=0.3359 @ epoch 19  ← best so far

Now testing longer windows: 1080h-2520h (45-105 days)

Configuration:
    - Resolution: Patch Size 21
    - Sensitivity: Flood Weight 8.0
    - Physics: Context Length 1080h+

Usage:
    python test_context_lengths.py --epochs 20 --context_lens 1080 1440 1800 2160 2520
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
DATA_FILE = "foundation_data_deep_105d.pkl"  # 105-day (2520h) history dataset
SCOUT_CHECKPOINT = "checkpoints/scout/scout_best.pth"
TIMERCD_PRETRAINED = "Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
PATCH_SIZE = 21  # Optimal from previous experiment


def weighted_reconstruction_loss(embeddings, targets, mask, model, flood_weight=8.0):
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


def evaluate(model, device, context_len, data_file=DATA_FILE, split='test'):
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
            
            # Future is after context_len
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


def train_with_context_len(context_len, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Train TimeRCDWithPrior with a specific context length."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"  Training with context_len = {context_len} ({context_len//24} days)")
    print(f"  patch_size = {PATCH_SIZE}")
    print(f"{'='*60}")
    
    # Load Dataset with specified context_len
    train_dataset = FloodDataset(DATA_FILE, split='train', context_len=context_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    config = TimeRCDConfig()
    config.ts_config.num_features = 1
    config.ts_config.d_model = 512
    config.ts_config.patch_size = PATCH_SIZE
    
    model = TimeRCDWithPrior(
        config, 
        scout_checkpoint=SCOUT_CHECKPOINT,
        context_len=context_len
    ).to(device)
    
    model.freeze_scout()
    
    # Load pretrained TimeRCD weights (filter incompatible layers)
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
        print(f"  Skipping {len(new_state_dict) - len(filtered_state_dict)} incompatible layers")
        
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
            loss = weighted_reconstruction_loss(embeddings, time_series, mask, model)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        metrics = evaluate(model, device, context_len)
        val_mccs.append(metrics['mcc'])
        
        if metrics['mcc'] > best_mcc:
            best_mcc = metrics['mcc']
            best_epoch = epoch + 1
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}")
    
    return {
        'context_len': context_len,
        'context_days': context_len // 24,
        'best_mcc': best_mcc,
        'best_epoch': best_epoch,
        'final_mcc': val_mccs[-1],
        'train_losses': train_losses,
        'val_mccs': val_mccs
    }


def run_experiments(context_lens, epochs=EPOCHS):
    """Run experiments for all context lengths and compare results."""
    print(f"\n{'#'*60}")
    print(f"  CONTEXT LENGTH EXPERIMENT (Tidal Physics)")
    print(f"  Testing: {context_lens} hours")
    print(f"  Days: {[c//24 for c in context_lens]}")
    print(f"  patch_size: {PATCH_SIZE} (fixed)")
    print(f"  Epochs per config: {epochs}")
    print(f"{'#'*60}")
    
    results = []
    
    for cl in context_lens:
        try:
            result = train_with_context_len(cl, epochs=epochs)
            results.append(result)
        except Exception as e:
            print(f"ERROR with context_len={cl}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'context_len': cl,
                'best_mcc': -1,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Context (h)':<12} {'Days':<8} {'Best MCC':<12} {'Best Epoch':<12} {'Final MCC':<12}")
    print("-" * 56)
    
    for r in results:
        if 'error' in r:
            print(f"{r['context_len']:<12} ERROR: {r['error'][:30]}")
        else:
            print(f"{r['context_len']:<12} {r['context_days']:<8} {r['best_mcc']:<12.4f} {r['best_epoch']:<12} {r['final_mcc']:<12.4f}")
    
    # Find best
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['best_mcc'])
        print(f"\n✅ BEST CONTEXT LENGTH: {best['context_len']}h ({best['context_days']} days)")
        print(f"   MCC={best['best_mcc']:.4f} at epoch {best['best_epoch']}")
    
    # Save results
    os.makedirs("experiment_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"experiment_results/context_len_exp_{timestamp}.json"
    
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
        label = f"{r['context_len']}h ({r['context_days']}d)"
        plt.plot(r['val_mccs'], label=label, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MCC')
    plt.title('MCC vs Epoch for Different Context Lengths')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    context_lens_plot = [r['context_len'] for r in valid_results]
    best_mccs_plot = [r['best_mcc'] for r in valid_results]
    labels = [f"{c}h\n({c//24}d)" for c in context_lens_plot]
    plt.bar(range(len(context_lens_plot)), best_mccs_plot, tick_label=labels)
    plt.xlabel('Context Length')
    plt.ylabel('Best MCC')
    plt.title('Best MCC by Context Length')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plot_file = f"experiment_results/context_len_exp_{timestamp}.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to: {plot_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test different context lengths for TimeRCD")
    parser.add_argument("--context_lens", type=int, nargs='+', default=[1080, 1440, 1800, 2160, 2520],
                        help="List of context lengths in hours (default: 1080 1440 1800 2160 2520)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs per configuration (default: 20)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (default: 256)")
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    run_experiments(args.context_lens, epochs=args.epochs)
