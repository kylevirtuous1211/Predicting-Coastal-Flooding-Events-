
import scipy.io
import pandas as pd
import numpy as np
import os
import pickle
import argparse
from datetime import datetime, timedelta

# Configuration
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "NEUSTG_19502020_12stations.mat")
THRESH_FILE = os.path.join(DATA_DIR, "Seed_Coastal_Stations_Thresholds.mat")

# Challenge Rules
TRAINING_STATIONS = ['Annapolis','Atlantic_City','Charleston','Washington','Wilmington', 'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook']
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

# Settings
PRED_WINDOW_DAYS = 14
HOURS_PER_DAY = 24
PRED_WINDOW = PRED_WINDOW_DAYS * HOURS_PER_DAY  # 336
STRIDE = 24  # Daily steps

def matlab2datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) \
           + timedelta(days=matlab_datenum % 1) \
           - timedelta(days=366)

def preprocess_data(hist_window_days=105):
    HIST_WINDOW = hist_window_days * HOURS_PER_DAY
    OUTPUT_FILE = f"foundation_data_deep_{hist_window_days}d.pkl"
    if not os.path.exists(DATA_FILE):
        print(f"Data file not found: {DATA_FILE}")
        return

    print("Loading raw data...")
    try:
        data_mat = scipy.io.loadmat(DATA_FILE)
        thresh_mat = scipy.io.loadmat(THRESH_FILE)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Extract Data
    sea_level = data_mat['sltg'] 
    station_names = [s[0] for s in data_mat['sname'].flatten()] 
    time = data_mat['t'].flatten()
    time_dt = pd.to_datetime([matlab2datetime(t) for t in time])
    latitudes = data_mat['lattg'].flatten()

    # Thresholds
    t_names = [s[0] for s in thresh_mat['sname'].flatten()]
    t_vals = thresh_mat['thminor_stnd'].flatten()
    threshold_map = dict(zip(t_names, t_vals))

    print(f"Processing stations with {HIST_WINDOW}h history...")
    
    train_data = []
    test_data = []
    all_data_values = []

    for i, name in enumerate(station_names):
        print(f"  Processing {name}...")
        thresh = threshold_map.get(name)
        lat = latitudes[i]
        
        raw_hourly = sea_level[:, i]
        valid_mask = ~np.isnan(raw_hourly)
        if valid_mask.sum() == 0:
            print(f"    Skipping {name} (No data)")
            continue

        # Interpolate small gaps
        df = pd.DataFrame({'time': time_dt, 'sea_level': raw_hourly})
        df['sea_level'] = df['sea_level'].interpolate(method='linear', limit=24)
        
        vals = df['sea_level'].values
        valid_mask = ~np.isnan(vals)
        
        if valid_mask.sum() == 0: continue

        # Normalization: (Value - Threshold) / Std
        station_std = np.nanstd(vals)
        if station_std == 0:
            station_std = 1.0
            
        norm_vals = (vals - thresh) / station_std
        
        num_hours = len(norm_vals)
        
        X_list = []
        Y_list = []
        
        for t in range(0, num_hours - HIST_WINDOW - PRED_WINDOW + 1, STRIDE):
            full_window = norm_vals[t : t + HIST_WINDOW + PRED_WINDOW]
            if np.isnan(full_window).any():
                continue
            
            x_win = norm_vals[t : t + HIST_WINDOW]
            y_win = norm_vals[t + HIST_WINDOW : t + HIST_WINDOW + PRED_WINDOW]
            
            X_list.append(x_win)
            Y_list.append(y_win)
            
        if not X_list:
            continue
            
        X_arr = np.array(X_list)  # (N, HIST_WINDOW)
        Y_arr = np.array(Y_list)  # (N, PRED_WINDOW)
        
        station_dict = {
            'name': name,
            'X': X_arr,
            'Y': Y_arr,
            'threshold': 0.0,
            'original_threshold': thresh,
            'std': station_std,
            'mean': np.nanmean(vals)
        }
        
        if name in TRAINING_STATIONS:
            train_data.append(station_dict)
        elif name in TESTING_STATIONS:
            test_data.append(station_dict)
            
        all_data_values.extend(norm_vals[~np.isnan(norm_vals)])

    global_mean = np.mean(all_data_values)
    global_std = np.std(all_data_values)
    
    print(f"Global Mean: {global_mean}, Global Std: {global_std}")
    
    # Summary
    total_train = sum(len(s['X']) for s in train_data)
    total_test = sum(len(s['X']) for s in test_data)
    print(f"Train samples: {total_train}, Test samples: {total_test}")
    print(f"X shape: ({HIST_WINDOW},), Y shape: ({PRED_WINDOW},)")
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({
            'train': train_data,
            'test': test_data,
            'global_mean': global_mean,
            'global_std': global_std
        }, f)
        
    print(f"Saved foundation data to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data with configurable history window")
    parser.add_argument("--hist_days", type=int, default=105,
                        help="History window in days (default: 105 = 2520 hours)")
    args = parser.parse_args()
    preprocess_data(hist_window_days=args.hist_days)
