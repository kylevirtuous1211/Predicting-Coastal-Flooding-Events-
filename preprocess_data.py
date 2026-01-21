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
    latitudes = data_mat['lattg'].flatten() # Needed for UTide

    # ... (Threshold extraction same as before) ...
    t_names = [s[0] for s in thresh_mat['sname'].flatten()]
    t_vals = thresh_mat['thminor_stnd'].flatten()
    threshold_map = dict(zip(t_names, t_vals))

    print("Processing stations...")
    
    # Store processed sequences
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    import utide # Import here or top level

    for i, name in enumerate(station_names):
        print(f"  Processing {name}...")
        thresh = threshold_map.get(name)
        lat = latitudes[i]
        
        # Get raw hourly data
        raw_hourly = sea_level[:, i]
        
        # Calculate Station Stats
        valid_mask = ~np.isnan(raw_hourly)
        valid_data = raw_hourly[valid_mask]
        station_std = np.std(valid_data)
        
        # UTide: Solve for constituents
        # We use the time in matplotlib date format or similar? 
        # utide expects time in days usually (datenum). 'time' variable is already Matlab datenum.
        
        print(f"    - Computing Astronomical Tide (UTide)...")
        coef = utide.solve(time[valid_mask], valid_data, lat=lat, verbose=False)
        
        # Reconstruct for entire timeline (including missing values)
        tide_reconst = utide.reconstruct(time, coef, verbose=False)
        tide_pred = tide_reconst['h']
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time_dt,
            'sea_level': raw_hourly,
            'tide_pred': tide_pred
        })
        
        # 1. Feature Engineering
        # A. Standardized Distance
        df['norm_val'] = (df['sea_level'] - thresh) / station_std
        
        # B. Surge (Residual)
        # The key physics-based feature
        df['surge'] = df['sea_level'] - df['tide_pred']
        
        # Normalize tide and surge too? 
        # XGBoost handles scale well, but normalizing helps convergence usually.
        # Let's keep them raw or normalize by station_std.
        df['norm_tide'] = df['tide_pred'] / station_std
        df['norm_surge'] = df['surge'] / station_std
        
        # C. Keep Seasonality? User said UTide replaces it, but Month/Day might capture thermal expansion not in tides.
        # Let's keep them.
        df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['time'].dt.dayofyear / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['time'].dt.dayofyear / 365.25)

        # Data Arrays
        # We need continuous arrays to slice windows
        # Using Normalized Surge and Tide as primary features
        norm_surge = df['norm_surge'].values
        norm_tide = df['norm_tide'].values
        
        # Seasonality (we only need it for the 'current' prediction time, i.e., end of input window)
        # Or maybe the middle? End is best (t=0 relative to prediction)
        month_sin = df['month_sin'].values
        month_cos = df['month_cos'].values
        doy_sin = df['doy_sin'].values
        doy_cos = df['doy_cos'].values
        
        raw_values = df['sea_level'].values
        
        num_hours = len(df)
        hours_per_window = HIST_WINDOW * 24 # 168
        hours_per_pred = PRED_WINDOW * 24   # 336
        
        station_X = []
        station_y = []
        
        # Slide by 24 hours (1 Day)
        # Range: Stop early enough to have full input AND full prediction
        # Input: [t : t+168]
        # Target: [t+168 : t+168+336]
        
        for t in range(0, num_hours - hours_per_window - hours_per_pred + 1, 24):
            # Input Window Indices: t to t+168
            # The 'current time' (predictions start) is at t+168
            
            # 1. Hourly Features (168 * 2 = 336 features)
            win_surge = norm_surge[t : t + hours_per_window]
            win_tide = norm_tide[t : t + hours_per_window]
            
            # Check for NaNs
            if np.isnan(win_surge).sum() > 24: 
                continue # Too many missing values
            win_surge = np.nan_to_num(win_surge)
            win_tide = np.nan_to_num(win_tide)
            
            # 2. Seasonality (Static for the window - take the last step i.e. "Now")
            idx_now = t + hours_per_window - 1
            seas_feats = np.array([
                month_sin[idx_now], month_cos[idx_now],
                doy_sin[idx_now], doy_cos[idx_now]
            ])
            
            # flatten and concat
            # [Surge_0...Surge_167, Tide_0...Tide_167, M_sin, M_cos, D_sin, D_cos]
            x_vec = np.concatenate([win_surge, win_tide, seas_feats])
            
            # Target
            y_window_raw = raw_values[t + hours_per_window : t + hours_per_window + hours_per_pred]
            y_days = y_window_raw.reshape(PRED_WINDOW, 24)
            y_daily_max = np.nanmax(y_days, axis=1)
            y_labels = (y_daily_max > thresh).astype(int)
            
            station_X.append(x_vec)
            station_y.append(y_labels)

        # 5. Assign
        if name in TRAINING_STATIONS:
            X_train.extend(station_X)
            y_train.extend(station_y)
        elif name in TESTING_STATIONS:
            X_val.extend(station_X)
            y_val.extend(station_y)
            
    # Save
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    print("\nProcessing Complete.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    # Feature Names logic:
    # 168 surge + 168 tide + 4 seas
    fnames = [f'surge_h{i}' for i in range(168)] + \
             [f'tide_h{i}' for i in range(168)] + \
             ['month_sin', 'month_cos', 'doy_sin', 'doy_cos']

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump({
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'feature_names': fnames
        }, f)
    print(f"Saved processed data to {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()
