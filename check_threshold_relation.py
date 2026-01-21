import scipy.io
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_FILE = r"data/NEUSTG_19502020_12stations.mat"
THRESH_FILE = r"data/Seed_Coastal_Stations_Thresholds.mat"

def analyze_relation():
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return

    data_mat = scipy.io.loadmat(DATA_FILE)
    thresh_mat = scipy.io.loadmat(THRESH_FILE)

    sea_level = data_mat['sltg']
    station_names = [s[0] for s in data_mat['sname'].flatten()]
    
    t_names = [s[0] for s in thresh_mat['sname'].flatten()]
    t_vals = thresh_mat['thminor_stnd'].flatten()
    threshold_map = dict(zip(t_names, t_vals))

    stats = []
    
    print(f"{'Station':<20} | {'Mean':<10} | {'Std':<10} | {'Thresh':<10} | {'Ratio (T/S)':<10} | {'Z-Score (T-M)/S':<10}")
    print("-" * 90)

    for i, name in enumerate(station_names):
        thresh = threshold_map.get(name)
        if thresh is None: continue

        raw_hourly = sea_level[:, i]
        valid_data = raw_hourly[~np.isnan(raw_hourly)]
        
        mean_sl = np.mean(valid_data)
        std_sl = np.std(valid_data)
        
        # How many std devs away is the threshold?
        z_score = (thresh - mean_sl) / std_sl
        
        stats.append({
            'name': name,
            'mean': mean_sl,
            'std': std_sl,
            'thresh': thresh,
            'z_score': z_score
        })
        
        print(f"{name:<20} | {mean_sl:.4f}     | {std_sl:.4f}     | {thresh:.4f}     | {thresh/std_sl:.4f}       | {z_score:.4f}")

    # Simple Regression
    df = pd.DataFrame(stats)
    print("\n--- Statistics ---")
    print(f"Average Z-Score (Threshold): {df['z_score'].mean():.4f}")
    print(f"Std Dev of Z-Score: {df['z_score'].std():.4f}")
    
    # Can we estimate Threshold ~ Mean + k * Std?
    print("\nEstimated Formula: Threshold approx Mean + (Avg_Z * Std)")
    
if __name__ == "__main__":
    analyze_relation()
