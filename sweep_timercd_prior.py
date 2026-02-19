import torch
import numpy as np
import os
import sys
import pickle
import argparse
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd # Ensure pandas is available

# Import directly from model_prior_token.py
# Assuming model_prior_token.py is in the current directory or python path
sys.path.append(os.getcwd())
try:
    from model_prior_token import TimeRCDWithPrior, TimeRCDConfig
except ImportError:
    print("Error: Could not import model_prior_token. Please ensure it is in the path.")
    sys.exit(1)

# Configuration Defaults (can be overridden)
DEFAULT_DATA_FILE = "foundation_data_deep_105d.pkl" 
DEFAULT_CHECKPOINT = "checkpoints/timercd_prior_FiLM/timercd_prior_best.pth"
CONTEXT_LEN = 1800
PATCH_SIZE = 21
BATCH_SIZE = 32

class FloodDataset(Dataset):
    def __init__(self, data_path, split='test', context_len=1800, pred_len=336):
        print(f"Loading data from {data_path} for split {split}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        if split not in data:
            raise ValueError(f"Split {split} not found in data file.")
            
        self.station_data = data[split]
        self.context_len = context_len
        self.pred_len = pred_len
        self.full_len = context_len + pred_len
        self.index_map = []
        
        print(f"Indexing samples (Context: {context_len})...")
        for s_idx, station in enumerate(self.station_data):
            num_samples = len(station['X'])
            for i in range(num_samples):
                # Ensure we have enough history if data structure varies, 
                # but assuming X is the history window. 
                # Actually X usually is fixed size in these pkls? 
                # Let's check size dynamically if needed, but for now standard logic:
                self.index_map.append((s_idx, i))
        print(f"Total samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        s_idx, local_idx = self.index_map[idx]
        item = self.station_data[s_idx]
        X = item['X'][local_idx]
        Y = item['Y'][local_idx]
        
        # X might be shorter than required context if not consistent
        # We need to handle that or assume data is prepared correctly.
        # Given 'foundation_data_deep_105d.pkl', it likely has long context.
        
        full_seq = np.concatenate([X, Y])
        
        # If full_seq is shorter than random context len, we pad? 
        # But here we assume data matches.
        
        # Fix for Shape Mismatch
        # foundation_data_deep_105d.pkl might have longer sequences (e.g. 2520 + 336 = 2856?)
        # We only need context_len + pred_len (1800 + 336 = 2136)
        
        target_len = self.full_len
        if len(full_seq) > target_len:
            # Take the LAST target_len points
            # because we want [History ... Future] 
            # and usually we predict the immediate future after history.
            full_seq = full_seq[-target_len:]
        elif len(full_seq) < target_len:
            # Pad if too short (prepend zeros)
            pad_len = target_len - len(full_seq)
            full_seq = np.pad(full_seq, (pad_len, 0))
            
        full_seq = torch.FloatTensor(full_seq).unsqueeze(-1)
        mask = torch.zeros(self.full_len, dtype=torch.bool)
        
        # Mask future (last pred_len)
        # 0 = History, 1 = Future (Masked)
        # We want to predict the future.
        mask[self.context_len:] = True 
        
        return {
            'time_series': full_seq, 
            'mask': mask, 
            'threshold': item.get('threshold', 0.0), 
            'station_name': item.get('name', 'unknown')
        }

def sweep(data_file, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found.")
        return
        
    test_dataset = FloodDataset(data_file, split='test', context_len=CONTEXT_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Initializing Model (TimeRCDWithPrior)...")
    config = TimeRCDConfig()
    config.ts_config.patch_size = PATCH_SIZE
    # We must match model dimensions.
    # d_model=512 is default in TimeRCDConfig
    
    try:
        model = TimeRCDWithPrior(config, context_len=CONTEXT_LEN).to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    print(f"Loading weights from {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            time_series = batch['time_series'].to(device)
            mask = batch['mask'].to(device)
            
            # Prepare Input: Zero out Future
            input_seq = time_series.clone()
            input_seq[mask.bool()] = 0.0
            
            # Attention mask for transformer (all ones usually, unless we want to mask padding)
            attention_mask = torch.ones((time_series.size(0), time_series.size(1)), dtype=torch.bool).to(device)
            
            # Forward pass through TimeRCDWithPrior
            # forward(time_series, mask, flood_prob=None)
            # We pass input_seq (with future zeroed) as time_series.
            # The model will use the history part of it for the Scout (if not provided flood_prob).
            embeddings = model(input_seq, attention_mask)
            
            # Reconstruction head
            reconstructed = model.reconstruction_head(embeddings)
            reconstructed = reconstructed.view(time_series.size(0), time_series.size(1), 1)
            
            for i in range(time_series.size(0)):
                future_mask = mask[i].bool()
                
                # Get Reconstructed Future
                future_preds = reconstructed[i][future_mask]
                
                # Score = Max value in prediction window
                score = future_preds.max().item()
                
                # Get Truth
                future_gt = time_series[i][future_mask]
                
                # Check for Flood
                # Assuming normalized > 0 means flood.
                label = (future_gt > 0).any().item()
                
                all_preds.append(score)
                all_labels.append(int(label))
                
    print("Scanning Thresholds...")
    # Thresholds for the *predicted score* (which is max normalized value)
    thresholds = np.arange(-5.0, 5.0, 0.1)
    
    results = []
    
    best_mcc = -1.0
    best_thresh = 0.0
    best_f1 = 0.0
    
    for t in thresholds:
        preds_bin = [1 if x > t else 0 for x in all_preds]
        mcc = matthews_corrcoef(all_labels, preds_bin)
        f1 = f1_score(all_labels, preds_bin, zero_division=0)
        
        results.append({'threshold': t, 'mcc': mcc, 'f1': f1})
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_thresh = t
            best_f1 = f1
            
    print(f"\n✅ BEST MCC: {best_mcc:.4f} at Threshold: {best_thresh:.2f}")
    print(f"✅ F1 Score: {best_f1:.4f}")
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("sweep_results_prior.csv", index=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['threshold'], df['mcc'], label='MCC')
    plt.plot(df['threshold'], df['f1'], label='F1')
    plt.axvline(x=best_thresh, color='r', linestyle='--', label=f'Best Thresh ({best_thresh:.2f})')
    plt.xlabel('Threshold (Normalized Sea Level)')
    plt.ylabel('Score')
    plt.title(f'Threshold Sweep Prior (Context: {CONTEXT_LEN}, Patch: {PATCH_SIZE})')
    plt.legend()
    plt.grid(True)
    plt.savefig('sweep_timercd_prior.png')
    print("Saved plot to sweep_timercd_prior.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_FILE)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    
    sweep(args.data, args.checkpoint)
