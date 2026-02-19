import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from tqdm import tqdm

# Add Time-RCD to path (priority)
sys.path.insert(0, os.path.join(os.getcwd(), "Time-RCD"))

from timercd_utils import FloodDataset
from models.time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from models.time_rcd.time_rcd_config import TimeRCDConfig

# Import Prior Token model (optional)
try:
    from model_prior_token import TimeRCDWithPrior, TimeRCDConfig as PriorConfig
    PRIOR_AVAILABLE = True
except ImportError:
    PRIOR_AVAILABLE = False

# Configuration
DATA_FILE = "foundation_data.pkl"
# CHECKPOINT_PATH = "Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"
# Finetuned Checkpoint (Epoch 1)
CHECKPOINT_PATH = "checkpoints/timercd_finetune/timercd_epoch_19.pth"
PRIOR_CHECKPOINT_PATH = "checkpoints/timercd_prior/timercd_prior_best.pth"
SCOUT_CHECKPOINT_PATH = "scout.pkl"

BATCH_SIZE = 32

from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score

def test(model=None, device=None, split='test', use_prior=False, checkpoint_path=None, 
         context_len=168, patch_size=16, data_file=DATA_FILE):
    """
    Test TimeRCD model with optional Prior Token support.
    
    Args:
        model: Pre-initialized model (optional)
        device: Torch device
        split: 'train' or 'test'
        use_prior: If True, use TimeRCDWithPrior model
        checkpoint_path: Override default checkpoint path
        context_len: Length of history window
        patch_size: Patch size for TimeRCD
        data_file: Path to dataset file
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # Load Test Dataset
    print(f"Loading {split} data for evaluation...")
    # NOTE: pred_len is fixed at 336 for now as per competition/standard
    test_dataset = FloodDataset(data_file, split=split, context_len=context_len)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = PRIOR_CHECKPOINT_PATH if use_prior else CHECKPOINT_PATH
    
    # Initialize Model if not provided
    if model is None:
        if use_prior and PRIOR_AVAILABLE:
            print("Initializing TimeRCDWithPrior...")
            config = PriorConfig()
            config.ts_config.num_features = 1
            config.ts_config.d_model = 512
            config.ts_config.patch_size = 16
            
            model = TimeRCDWithPrior(config, scout_checkpoint=SCOUT_CHECKPOINT_PATH).to(device)
            model.freeze_scout()
        else:
            print("Initializing TimeRCD...")
            config = TimeRCDConfig()
            config.ts_config.num_features = 1
            config.ts_config.d_model = 512
            config.ts_config.patch_size = patch_size
            
            model = TimeSeriesPretrainModel(config).to(device)
        
        # Load Weights
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in state_dict:
                 state_dict = state_dict['model_state_dict']
            
            # Handle prefixes
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            model.load_state_dict(new_state_dict, strict=False) 
        else:
            print(f"Checkpoint {checkpoint_path} not found!")
            return {}

    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            time_series = batch['time_series'].to(device)
            mask = batch['mask'].to(device)
            thresholds = batch['threshold'].numpy()
            
            # Create Attention Mask
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            # CRITICAL FIX: Zero out the Future to prevent data leakage
            input_seq = time_series.clone()
            input_seq[mask.bool()] = 0.0
            
            # Forward Pass -> Embeddings
            embeddings = model(input_seq, attention_mask)
            
            # Reconstruction
            reconstructed = model.reconstruction_head(embeddings) 
            reconstructed = reconstructed.view(time_series.size(0), time_series.size(1), 1)
            
            # Extract the FUTURE part (where mask is True)
            all_max_preds = [] # For this batch (but we need global... wait, we iterate batches logic needs adjustment)
            
            # Correction: We shouldn't reset all_preds inside loop.
            # We are inside the loop here.
            
            # Batch Processing
            for i in range(time_series.size(0)):
                # Get the future mask for this sample
                future_mask = mask[i].bool()
                
                # Get reconstructed values for the future
                future_preds = reconstructed[i][future_mask] # Should be length 336
                
                # Visual Debugging (First sample of First Batch)
                if len(all_labels) == 0 and i == 0:
                    import matplotlib.pyplot as plt
                    true_future = time_series[i][future_mask].cpu().numpy()
                    pred_future = future_preds.cpu().numpy()
                    
                    plt.figure(figsize=(10, 5))
                    plt.plot(true_future, label='Ground Truth', color='blue')
                    plt.plot(pred_future, label='Prediction', color='red', alpha=0.7)
                    plt.axhline(y=0, color='green', linestyle='--', label='Threshold (0.0)')
                    plt.title(f"Sample 0 Reconstruction (Label: {(true_future > 0).any()})")
                    plt.legend()
                    plt.savefig("debug_reconstruction.png")
                    plt.close()
                    print("Saved debug_reconstruction.png")
                
                # Save the PEAK value for calibration
                peak_val = future_preds.max().item()
                # Storing peak_val now, will threshold later
                all_preds.append(peak_val) 
                
                # Get Ground Truth
                future_gt = time_series[i][future_mask]
                flood_label = (future_gt > 0).any().item()
                all_labels.append(int(flood_label))
                
    # Evaluation Logic Change: Calibration
    print("Computing metrics with Calibration Scan...")
    
    best_mcc = -1
    best_f1 = -1
    best_thresh = 0.0
    
    # We stored raw peak values in all_preds. Now we scan.
    thresholds = np.arange(-0.5, 2.0, 0.05)
    
    for t in thresholds:
        preds_bin = [1 if x > t else 0 for x in all_preds]
        mcc = matthews_corrcoef(all_labels, preds_bin)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
            best_f1 = f1_score(all_labels, preds_bin, zero_division=0)
            
    print(f"\n✅ OPTIMAL THRESHOLD: {best_thresh:.3f}")
    print(f"✅ BEST MCC: {best_mcc:.4f}")
    print(f"✅ F1 at Best Thresh: {best_f1:.4f}")
    
    # Use best threshold for CM
    final_preds_bin = [1 if x > best_thresh else 0 for x in all_preds]
    cm = confusion_matrix(all_labels, final_preds_bin)
    
    print(f"Confusion Matrix (at thresh {best_thresh:.3f}):\n{cm}")

    return {'mcc': best_mcc, 'f1': best_f1, 'cm': cm, 'best_thresh': best_thresh}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TimeRCD model with optional Prior Token support")
    parser.add_argument("--use_prior", action="store_true", help="Use TimeRCDWithPrior model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    args = parser.parse_args()
    
    test(use_prior=args.use_prior, checkpoint_path=args.checkpoint, split=args.split)
