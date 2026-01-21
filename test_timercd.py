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

from timercd_utils import FloodDataset
from models.time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from models.time_rcd.time_rcd_config import TimeRCDConfig

# Configuration
DATA_FILE = "foundation_data.pkl"
# CHECKPOINT_PATH = "checkpoints/timercd_finetune/timercd_epoch_1.pth" # Or best
# Zero-shot evaluation using pretrained model:
CHECKPOINT_PATH = "Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"

BATCH_SIZE = 32

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Test Dataset
    print("Loading test data...")
    test_dataset = FloodDataset(DATA_FILE, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    print("Initializing TimeRCD...")
    config = TimeRCDConfig()
    config.ts_config.num_features = 1
    config.ts_config.d_model = 512
    config.ts_config.patch_size = 16
    
    model = TimeSeriesPretrainModel(config).to(device)
    
    # Load Weights
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}")
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
                
        model.load_state_dict(new_state_dict, strict=False) 
    else:
        print(f"Checkpoint {CHECKPOINT_PATH} not found!")
        return

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
            
            # Forward Pass -> Embeddings
            embeddings = model(time_series, attention_mask)
            
            # Reconstruction
            # We need to manually call the reconstruction head since masked_reconstruction_loss computes loss directly
            # Model structure (TimeRCD_pretrain_multi.py):
            # self.reconstruction_head = nn.Sequential(...)
            
            reconstructed = model.reconstruction_head(embeddings) # (B, 504, 1, 1) -> Wait, let's check shape
            # In TimeRCD_pretrain_multi.py:
            # reconstructed = self.reconstruction_head(local_embeddings)  # (B, seq_len, num_features, 1)
            # reconstructed = reconstructed.view(batch_size, seq_len, num_features)
            
            reconstructed = reconstructed.view(time_series.size(0), time_series.size(1), 1)
            
            # Extract the FUTURE part (where mask is True)
            # mask is (B, 504).
            
            # We want to check if ANY point in the future prediction > threshold
            # Since data is normalized (value - threshold)/std, the threshold is 0.0
            
            # Iterate over batch to handle masks correctly
            for i in range(time_series.size(0)):
                # Get the future mask for this sample
                # In FloodDataset, mask is 0 for history, 1 for future
                future_mask = mask[i].bool()
                
                # Get reconstructed values for the future
                future_preds = reconstructed[i][future_mask] # Should be length 336
                
                # Check for flooding (Any value > 0)
                # Note: Normalized data. 0 corresponds to original threshold.
                flood_pred = (future_preds > 0).any().item()
                all_preds.append(int(flood_pred))
                
                # Get Ground Truth
                # We need the original values to determine label?
                # The input `time_series` contains ground truth for the future part too (it was concatenated X+Y)
                future_gt = time_series[i][future_mask]
                flood_label = (future_gt > 0).any().item()
                all_labels.append(int(flood_label))
                
    # Evaluation
    print("Computing metrics...")
    cm = confusion_matrix(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    print(f"Confusion Matrix:\n{cm}")
    print(f"MCC: {mcc:.4f}")

if __name__ == "__main__":
    test()
