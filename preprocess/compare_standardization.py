import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
DATA_DIR = r"c:\Users\Admin-62501\Desktop\Coding\Projects\Predicting-Coastal-Flooding-Events-\data"
DATA_FILE = os.path.join(DATA_DIR, "NEUSTG_19502020_12stations.mat")
THRESH_FILE = os.path.join(DATA_DIR, "Seed_Coastal_Stations_Thresholds.mat")
OUTPUT_DIR = "analysis_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compare_standardization():
    print("Loading data...")
    try:
        data_mat = scipy.io.loadmat(DATA_FILE)
        thresh_mat = scipy.io.loadmat(THRESH_FILE)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    sea_level = data_mat['sltg'] 
    station_names = [s[0] for s in data_mat['sname'].flatten()] 
    
    t_names = [s[0] for s in thresh_mat['sname'].flatten()]
    t_vals = thresh_mat['thminor_stnd'].flatten()
    threshold_map = dict(zip(t_names, t_vals))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
    
    # 1. Just Threshold Subtraction
    ax1 = axes[0]
    for i, name in enumerate(station_names):
        thresh = threshold_map.get(name)
        data = sea_level[:, i]
        clean_data = data[~np.isnan(data)]
        
        norm_data = clean_data - thresh
        ax1.hist(norm_data, bins=50, density=True, histtype='step', label=name)

    ax1.set_title("1. Distance to Threshold (X - Threshold)")
    ax1.set_xlabel("Meters relative to Threshold")
    ax1.axvline(0, color='black', linestyle='--')
    ax1.grid(True, alpha=0.3)

    # 2. Z-Score (ignoring threshold for a moment, just standard scaling)
    ax2 = axes[1]
    for i, name in enumerate(station_names):
        data = sea_level[:, i]
        clean_data = data[~np.isnan(data)]
        
        mean = np.mean(clean_data)
        std = np.std(clean_data)
        
        z_data = (clean_data - mean) / std
        ax2.hist(z_data, bins=50, density=True, histtype='step', label=name)

    ax2.set_title("2. Standard Z-Score ((X - Mean) / Std)")
    ax2.set_xlabel("Standard Deviations from Mean")
    ax2.grid(True, alpha=0.3)

    # 3. Hybrid: (X - Threshold) / Std 
    # This scales the "Distance to Threshold" by the volatility of that station
    ax3 = axes[2]
    for i, name in enumerate(station_names):
        thresh = threshold_map.get(name)
        data = sea_level[:, i]
        clean_data = data[~np.isnan(data)]
        
        std = np.std(clean_data)
        
        # How many Std Devs away from the Flood Threshold are we?
        hybrid_data = (clean_data - thresh) / std
        ax3.hist(hybrid_data, bins=50, density=True, histtype='step', label=name)

    ax3.set_title("3. Standardized Distance ((X - Threshold) / Std)")
    ax3.set_xlabel("Standard Deviations from Threshold (0 = Flood)")
    ax3.axvline(0, color='black', linestyle='--')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "standardization_comparison.png")
    plt.savefig(plot_path)
    print(f"Saved comparison to {plot_path}")

if __name__ == "__main__":
    compare_standardization()
