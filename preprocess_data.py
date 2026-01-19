import scipy.io
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

# Configuration
DATA_DIR = r"c:\Users\Admin-62501\Desktop\Coding\Projects\Predicting-Coastal-Flooding-Events-\data"
DATA_FILE = os.path.join(DATA_DIR, "NEUSTG_19502020_12stations.mat")
THRESH_FILE = os.path.join(DATA_DIR, "Seed_Coastal_Stations_Thresholds.mat")
OUTPUT_FILE = "processed_data.pkl"

# Challenge Rules
TRAINING_STATIONS = ['Annapolis','Atlantic_City','Charleston','Washington','Wilmington', 'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook']
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

HIST_WINDOW = 7
PRED_WINDOW = 14

def matlab2datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) \
           + timedelta(days=matlab_datenum % 1) \
           - timedelta(days=366)

def preprocess_data():
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
    
    # Extract Thresholds
    t_names = [s[0] for s in thresh_mat['sname'].flatten()]
    t_vals = thresh_mat['thminor_stnd'].flatten()
    threshold_map = dict(zip(t_names, t_vals))

    print("Processing stations...")
    
    # Store processed sequences
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    for i, name in enumerate(station_names):
        print(f"  Processing {name}...")
        thresh = threshold_map.get(name)
        
        # Get raw hourly data
        raw_hourly = sea_level[:, i]
        
        # Calculate Station Stats (using entire history for climatology)
        valid_data = raw_hourly[~np.isnan(raw_hourly)]
        station_std = np.std(valid_data)
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time_dt,
            'sea_level': raw_hourly
        })
        
        # 1. Feature Engineering: Standardized Distance
        # feature = (sea_level - threshold) / station_std
        df['norm_val'] = (df['sea_level'] - thresh) / station_std
        
        # 2. Aggregating to Daily
        df_daily = df.set_index('time').resample('D').agg({
            'sea_level': 'max',  # Max raw level for checking flood
            'norm_val': ['mean', 'max', 'min', 'std'] # Features
        })
        
        # Flatten MultiIndex columns
        df_daily.columns = ['_'.join(col).strip() for col in df_daily.columns.values]
        
        # Rename for clarity
        df_daily = df_daily.rename(columns={
            'sea_level_max': 'max_raw_sl',
            'norm_val_mean': 'feat_mean',
            'norm_val_max':  'feat_max',
            'norm_val_min':  'feat_min',
            'norm_val_std':  'feat_std'
        })
        
        # 3. Create Binary Target
        # Flood = 1 if max_raw_sl > threshold
        df_daily['target'] = (df_daily['max_raw_sl'] > thresh).astype(int)
        
        # Drop NaNs (days with missing data)
        df_daily = df_daily.dropna()
        
        # 4. Create Sliding Windows
        # Input: 7 days of features
        # Output: 14 days of targets
        
        feature_cols = ['feat_mean', 'feat_max', 'feat_min', 'feat_std']
        data_values = df_daily[feature_cols].values
        target_values = df_daily['target'].values
        
        num_samples = len(df_daily) - HIST_WINDOW - PRED_WINDOW
        
        station_X = []
        station_y = []
        
        if num_samples > 0:
            for t in range(0, num_samples, 1):
                # Input: [t : t+7]
                x_window = data_values[t : t+HIST_WINDOW].flatten()
                # Target: [t+7 : t+7+14]
                y_window = target_values[t+HIST_WINDOW : t+HIST_WINDOW+PRED_WINDOW]
                
                station_X.append(x_window)
                station_y.append(y_window)
        
        # 5. Assign to Train or Val depending on station name
        if name in TRAINING_STATIONS:
            X_train.extend(station_X)
            y_train.extend(station_y)
        elif name in TESTING_STATIONS:
            X_val.extend(station_X)
            y_val.extend(station_y)
        else:
            print(f"Warning: Station {name} not in Train or Test lists. Skipping.")

    # Convert to arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    print("\nProcessing Complete.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # Save
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'feature_names': feature_cols
        }, f)
    print(f"Saved processed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()
