import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import pandas as pd
from datetime import datetime, timedelta
from scipy.io import loadmat
from tqdm import tqdm

# Add current dir to path
sys.path.append(os.getcwd())
try:
    from model_timercd import TimeSeriesPretrainModel, TimeRCDConfig
except ImportError:
    # If copied/moved, try to find it
    sys.path.append("/home/cvlab123/Kyle_Having_Fun/NSF_HDR_Year2/Predicting-Coastal-Flooding-Events-")
    from model_timercd import TimeSeriesPretrainModel, TimeRCDConfig

# Configuration
CHECKPOINT = "checkpoints/timercd_finetune/75days/timercd_epoch_60.pth"
MAT_FILE = "./data/NEUSTG_19502020_12stations.mat"
METADATA_FILE = "station_metadata.pkl"
INTERVALS_FILE = "official_scripts/seed_official_time_intervals.txt"
CONTEXT_LEN = 1800
PATCH_SIZE = 21
THRESHOLD = -0.10
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

def matlab2datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=float(matlab_datenum) % 1) - timedelta(days=366)

def load_official_data():
    print(f"Loading official MAT data from {MAT_FILE}...")
    d = loadmat(MAT_FILE)
    station_names = [s[0] for s in d['sname'].flatten()]
    time = d['t'].flatten()
    time_dt = np.array([matlab2datetime(t) for t in time])
    sea_level = d['sltg'] # (T, S)
    
    station_data = {}
    for i, name in enumerate(station_names):
        vals = sea_level[:, i]
        # Interpolate NaNs (Match ingestion.py logic)
        vals_series = pd.Series(vals)
        if vals_series.isnull().any():
            print(f"  Fixing NaNs in {name}...")
            vals = vals_series.interpolate(limit=24).fillna(method='bfill').fillna(method='ffill').values
        
        station_data[name] = {
            'values': vals,
            'times': time_dt
        }
    return station_data

def load_metadata():
    with open(METADATA_FILE, 'rb') as f:
        return pickle.load(f)

def parse_intervals():
    intervals = []
    with open(INTERVALS_FILE, 'r') as f:
        lines = f.readlines()[1:] # Skip header
        for line in lines:
            if not line.strip(): continue
            start, end = line.strip().split('\t')
            intervals.append({
                'start': datetime.strptime(start, "%m/%d/%Y"),
                'end': datetime.strptime(end, "%m/%d/%Y")
            })
    return intervals

def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    config = TimeRCDConfig()
    config.ts_config.patch_size = PATCH_SIZE
    # Ensure d_model matches (it was updated in model_timercd.py ingestion_predict too)
    config.ts_config.d_model = 512 
    
    model = TimeSeriesPretrainModel(config).to(device)
    
    if os.path.exists(CHECKPOINT):
        print(f"Loading checkpoint {CHECKPOINT}")
        state_dict = torch.load(CHECKPOINT, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded.")
    else:
        print(f"Checkpoint {CHECKPOINT} not found!")
        # return # Proceed anyway to see if script runs? No, need weights.
        sys.exit(1)

    model.eval()
    
    station_data = load_official_data()
    metadata = load_metadata()
    intervals = parse_intervals()
    
    os.makedirs("visualizations_official", exist_ok=True)
    
    print("Generating visualizations for official intervals...")
    with torch.no_grad():
        for st_name in TESTING_STATIONS:
            if st_name not in station_data:
                print(f"Warning: Station {st_name} not found in MAT file.")
                continue
            
            st_full_data = station_data[st_name]
            st_meta = metadata.get(st_name)
            if not st_meta:
                print(f"Warning: No metadata for {st_name}. Estimating...")
                mean = np.mean(st_full_data['values'])
                std = np.std(st_full_data['values'])
                thresh = mean + 2.29 * std
            else:
                mean = st_meta['mean']
                std = st_meta['std']
                thresh = st_meta['threshold']
            
            times = st_full_data['times']
            # Normalize full series
            norm_values = (st_full_data['values'] - thresh) / std
            
            for idx, interval in enumerate(intervals):
                # The interval end_date in the txt is inclusive for 7 days.
                # Ingestion script says hist_start = anchor - 7 days.
                # For visualization, let's treat interval['end'] as the 'anchor' (start of prediction)
                # But looking at intervals: 3/6/1962 to 3/12/1962 is 7 days. 
                # Prediction starts on 3/13/1962.
                
                anchor_date = interval['end'] + timedelta(days=1)
                
                # Find index of anchor_date in times
                try:
                    anchor_idx = np.where(times >= anchor_date)[0][0]
                except IndexError:
                    print(f"Interval {idx} for {st_name} out of bounds.")
                    continue
                
                # We need CONTEXT_LEN (1800h) for the model, but user wants to SEE 7 days.
                # We also predict 336h (14 days).
                
                start_idx = anchor_idx - CONTEXT_LEN
                end_idx = anchor_idx + 336
                
                if start_idx < 0 or end_idx > len(norm_values):
                    print(f"Interval {idx} for {st_name} has insufficient padding.")
                    continue
                
                seq = norm_values[start_idx:end_idx]
                gt_seq = seq.copy()
                
                # Prepare Model Input (1800h context)
                input_seq = torch.FloatTensor(seq[:CONTEXT_LEN]).view(1, CONTEXT_LEN, 1).to(device)
                # Pad for patch size if needed (though 1800 is div by 21? 1800/21 = 85.7... No)
                # Actually model handles padding.
                
                # Zero out future for prediction? No, model only looks at history passed.
                # But my model forward pass takes (time_series, mask).
                # Actually TimeSeriesPretrainModel(config).forward(input_seq, attention_mask)
                # In model_timercd.py: 
                # embeddings = model(time_series, attention_mask)
                # reconstructed = model.reconstruction_head(embeddings)
                
                # Let's use the full 2136 length (1800 + 336) and mask the future.
                full_input = torch.zeros((1, CONTEXT_LEN + 336, 1)).to(device)
                full_input[0, :CONTEXT_LEN, 0] = input_seq[0, :, 0] # [B, Time, Channel]

                # User wants 7 days history. 7*24 = 168.
                plot_hist_start = CONTEXT_LEN - 168
                
                # Option 1: Full attention (match ingestion_predict)
                mask_ones = torch.ones((1, CONTEXT_LEN + 336), dtype=torch.bool).to(device)
                
                embeddings = model(full_input, mask_ones)
                reconstructed = model.reconstruction_head(embeddings).view(-1).cpu().numpy()
                
                # Debugging Stats
                future_pred = reconstructed[CONTEXT_LEN:CONTEXT_LEN+336]
                hist_pred = reconstructed[plot_hist_start:CONTEXT_LEN]
                
                if np.isnan(future_pred).any():
                    print(f"  [ERROR] Interval {idx+1} for {st_name} has NaNs in prediction!")
                else:
                    h_std = np.std(hist_pred)
                    f_std = np.std(future_pred)
                    print(f"  Interval {idx+1} for {st_name}: Future Min={future_pred.min():.2f}, Max={future_pred.max():.2f} | Hist Std={h_std:.3f}")
                
                # Plotting
                plt.figure(figsize=(14, 7))
                
                # Indices for plotting
                # History (7 days)
                hist_idx_plot = np.arange(-168, 0)
                # 1. 7-day window data (black line)
                plt.plot(hist_idx_plot, gt_seq[plot_hist_start:CONTEXT_LEN], color='black', linewidth=2, label='History (7 Days)')
                
                # 1b. Reconstruction of History (to see if model is jittery in-sample)
                plt.plot(hist_idx_plot, hist_pred, color='blue', linewidth=1, alpha=0.6, label='History Reconstruction')
                
                # 2. Next 14 days ground truth (dotted black line)
                future_idx_plot = np.arange(0, 336)
                plt.plot(future_idx_plot, gt_seq[CONTEXT_LEN:CONTEXT_LEN+336], color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Ground Truth (14 Days)')
                plt.plot(future_idx_plot, reconstructed[CONTEXT_LEN:CONTEXT_LEN+336], color='red', linewidth=2, label='Prediction (14 Days)')
                
                # Add threshold and zero reference
                plt.axhline(y=THRESHOLD, color='green', linestyle=':', label=f'Threshold ({THRESHOLD})')
                plt.axhline(y=0, color='blue', linestyle='-.', alpha=0.2)
                
                # Formatting
                plt.title(f"TimeRCD Official Interval {idx+1}: {st_name}\n({interval['start'].date()} to {interval['end'].date()} Seed)", fontsize=14)
                plt.xlabel("Hours from Prediction Reference", fontsize=12)
                plt.ylabel("Normalized Sea Level", fontsize=12)
                plt.legend(loc='upper right')
                plt.grid(True, which='both', linestyle='--', alpha=0.5)
                plt.xlim(-168, 336)
                
                # Highlighting
                plt.axvspan(0, 336, color='yellow', alpha=0.1, label='Prediction Period')
                
                plt.tight_layout()
                save_name = f"visualizations_official/{st_name}_interval_{idx+1}.png"
                plt.savefig(save_name)
                plt.close()
                
    print("\nGeneration complete. Plots saved in 'visualizations_official/' directory.")

if __name__ == "__main__":
    visualize()
