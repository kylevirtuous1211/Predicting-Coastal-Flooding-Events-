import pickle
import numpy as np
import os
import sys
from tqdm import tqdm

# Add Time-RCD to path (priority)
sys.path.insert(0, os.path.join(os.getcwd(), "Time-RCD"))

from timercd_utils import FloodDataset

DATA_FILE = "foundation_data.pkl"

def analyze_split(split):
    print(f"Analyzing {split} split...")
    dataset = FloodDataset(DATA_FILE, split=split)
    
    total_samples = len(dataset)
    anomaly_samples = 0
    normal_samples = 0
    
    # We can iterate through the dataset
    # But since FloodDataset loads everything into memory in __init__, we can access directly if we want speed
    # dataset.station_data is list of dicts.
    
    # Let's iterate properly through getitem to be safe with logic
    for i in tqdm(range(total_samples)):
        item = dataset[i]
        time_series = item['time_series'] # (504, 1) or (504, 3)
        mask = item['mask'] # (504,)
        
        # Future part
        future_mask = mask.bool()
        
        # Determine if it's 3 channel or 1 channel
        if time_series.shape[-1] == 3:
            future_vals = time_series[future_mask, 0] # Sea Level is channel 0
        else:
            future_vals = time_series[future_mask, 0]
            
        # Check anomaly
        # Normalized data: > 0 means Flood (Anomaly)
        if (future_vals > 0).any():
            anomaly_samples += 1
        else:
            normal_samples += 1
            
    print(f"--- {split.upper()} RESULTS ---")
    print(f"Total Samples: {total_samples}")
    print(f"Anomaly Samples (Flood): {anomaly_samples} ({anomaly_samples/total_samples*100:.2f}%)")
    print(f"Normal Samples (No Flood): {normal_samples} ({normal_samples/total_samples*100:.2f}%)")
    
    # Calculate global statistics
    # This might be slow if we iterate, but it's safe.
    # We can sample or accumulate sum/sum_sq.
    
    all_vals = []
    # Sampling for speed if dataset is huge, but let's try full pass first or just reuse the loop above if we modify it.
    # Let's modify the loop to collect values.
    
    # Re-looping isn't ideal. Let's do it in the main loop but avoiding storing all 70k * 504 floats in a list if possible.
    # Actually 70k * 504 floats is ~140MB, totally fine.
    
    # Re-write the metrics calc to be efficient inside the loop
    pass

def analyze_split_fast(split):
    print(f"Analyzing {split} split...")
    dataset = FloodDataset(DATA_FILE, split=split)
    
    total_samples = len(dataset)
    anomaly_samples = 0
    
    sum_val = 0.0
    sum_sq_val = 0.0
    count_val = 0
    
    for i in tqdm(range(total_samples)):
        item = dataset[i]
        time_series = item['time_series'] # (504, 1)
        
        # Check flood
        future_mask = item['mask'].bool()
        future_vals = time_series[future_mask, 0]
        if (future_vals > 0).any():
            anomaly_samples += 1
            
        # Stats on the entire sequence (Context + Future) or just Context?
        # Usually we care about the input distribution.
        # Let's do Full Sequence.
        vals = time_series[:, 0].numpy()
        sum_val += vals.sum()
        sum_sq_val += (vals ** 2).sum()
        count_val += len(vals)
        
    mean = sum_val / count_val
    std = np.sqrt((sum_sq_val / count_val) - (mean ** 2))
    
    normal_samples = total_samples - anomaly_samples
    
    print(f"--- {split.upper()} RESULTS ---")
    print(f"Total Samples: {total_samples}")
    print(f"Anomaly Rate: {anomaly_samples/total_samples*100:.2f}%")
    print(f"Pixel Mean: {mean:.4f}")
    print(f"Pixel Std:  {std:.4f}")
    print("--------------------------\n")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
    else:
        analyze_split_fast('train')
        analyze_split_fast('test')
