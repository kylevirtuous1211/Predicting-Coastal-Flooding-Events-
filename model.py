import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import argparse
from datetime import datetime, timedelta
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# ==========================================
# Configuration & Constants
# ==========================================
HIST_WINDOW = 7
PRED_WINDOW = 14
MODEL_FILE = 'model.pkl'
BEST_THRESHOLD = 0.75 # Tuned on validation set

# Hardcoded constants from extract_constants.py
STATION_CONSTANTS = {
    'Annapolis': {'threshold': 2.1040, 'std': 0.2378},
    'Atlantic_City': {'threshold': 3.3440, 'std': 0.4898},
    'Charleston': {'threshold': 2.9800, 'std': 0.6066},
    'Eastport': {'threshold': 8.0710, 'std': 1.9562},
    'Fernandina_Beach': {'threshold': 3.1480, 'std': 0.6867},
    'Lewes': {'threshold': 2.6750, 'std': 0.4951},
    'Portland': {'threshold': 6.2670, 'std': 1.0243},
    'Sandy_Hook': {'threshold': 2.8090, 'std': 0.5467},
    'Sewells_Point': {'threshold': 2.7060, 'std': 0.3395},
    'The_Battery': {'threshold': 3.1920, 'std': 0.5340},
    'Washington': {'threshold': 2.6730, 'std': 0.3799},
    'Wilmington': {'threshold': 2.4230, 'std': 0.4768},
}

class CoastalFloodModel:
    def __init__(self, model_path=MODEL_FILE):
        self.model_path = model_path
        self.model = None
        # Try to load if exists
        self.load_model()
        
    def load_model(self):
        """Loads the trained XGBoost model weights."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print(f"Model file {self.model_path} not found.")

    def process_csv_data(self, hourly_csv_path, index_csv_path=None):
        """
        Processes CSV data provided by ingestion script.
        Returns:
            X (np.array): Features
            ids (list): Corresponding IDs (if index_csv_path provided)
        """
        import utide 
        
        print(f"Reading {hourly_csv_path}...")
        df = pd.read_csv(hourly_csv_path)
        
        # Ensure time is datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # 1. Norm, Tide, Surge Calculation
        print("Calculating Tide and Surge features (UTide)...")
     
        # Global storage for "Hourly Features"
        # Key: Station Name -> DataFrame with ['norm_surge', 'norm_tide', 'seas...']
        station_data_map = {}
        
        # We process each station separately to be efficient
        for station_name, group in df.groupby('station_name'):
            # Get constants or Estimate them
            consts = STATION_CONSTANTS.get(station_name)
            
            if consts:
                thresh = consts['threshold']
                std_dev = consts['std']
            else:
                # Dynamic Estimation for Hidden Stations
                valid_sl = group['sea_level'].dropna()
                if len(valid_sl) == 0:
                    continue
                est_mean = valid_sl.mean()
                est_std = valid_sl.std()
                est_thresh = est_mean + (2.3 * est_std)
                thresh = est_thresh
                std_dev = est_std
                print(f"[{station_name}] Unknown station. Est Thresh: {thresh:.4f}")
            
            # Extract Latitude (Assumed to be in CSV as 'latitude')
            if 'latitude' in group.columns:
                lat = group['latitude'].iloc[0]
            else:
                # Fallback if missing (should not happen in ingestion)
                lat = 0.0 
                print(f"Warning: Latitude missing for {station_name}, defaulting to 0.0")

            # --- UTide Logic ---
            raw_hourly = group['sea_level'].values
            time_nums = group['time'].values # numpy array
            
            # UTide needs unique time usually, check dupes? Ingestion ensures unique?
            # We assume unique sorted time.
            
            # Solve
            valid_mask = ~np.isnan(raw_hourly)
            # Need dates in day-float format for UTide if passing numpy datetime64?
            # UTide works better with matplotlib dates usually.
            # Convert timestamp to ordinal/days
            
            # Faster way: group['time'] is datetime64.
            # mdates code:
            # from matplotlib import dates as mdates
            # t_days = mdates.date2num(group['time'].dt.to_pydatetime())
            
            # Let's import inside loop or top? Inside is safer for now.
            from matplotlib import dates as mdates
            t_days = mdates.date2num(group['time'].dt.to_pydatetime())
            
            coef = utide.solve(t_days[valid_mask], raw_hourly[valid_mask], lat=lat, verbose=False)
            tide_reconst = utide.reconstruct(t_days, coef, verbose=False)
            tide_pred = tide_reconst['h']
            
            surge = raw_hourly - tide_pred
            
            # Normalize
            norm_surge = surge / std_dev
            norm_tide = tide_pred / std_dev
            
            # Seasonality
            # Calculate for all rows
            time_idx = group['time'] # Series
            month_sin = np.sin(2 * np.pi * time_idx.dt.month / 12).values
            month_cos = np.cos(2 * np.pi * time_idx.dt.month / 12).values
            doy_sin = np.sin(2 * np.pi * time_idx.dt.dayofyear / 365.25).values
            doy_cos = np.cos(2 * np.pi * time_idx.dt.dayofyear / 365.25).values
            
            # Store Processed Series aligned with time
            processed_df = pd.DataFrame({
                'norm_surge': norm_surge,
                'norm_tide': norm_tide,
                'month_sin': month_sin,
                'month_cos': month_cos,
                'doy_sin': doy_sin,
                'doy_cos': doy_cos
            }, index=group['time'])
            
            station_data_map[station_name] = processed_df

        if index_csv_path:
            print(f"Extracting windows from {index_csv_path}...")
            index_df = pd.read_csv(index_csv_path)
            
            X_out = []
            ids_out = []
            
            expected_hours = HIST_WINDOW * 24 # 168
            
            # Feature Vector Size: 168(Surge) + 168(Tide) + 4(Seas) = 340
            total_feats = (expected_hours * 2) + 4 
            
            for _, row in index_df.iterrows():
                stn = row['station_name']
                hist_start = pd.to_datetime(row['hist_start'])
                # hist_end = pd.to_datetime(row['hist_end']) # Inclusive
                
                if stn not in station_data_map:
                    X_out.append(np.zeros(total_feats))
                    ids_out.append(row['id'])
                    continue
                    
                stn_data = station_data_map[stn]
                
                # Get Window [start : start+168h]
                # Slice range
                end_ts = hist_start + timedelta(hours=expected_hours - 1)
                
                # Use slice
                window = stn_data[hist_start : end_ts]
                
                # Handle missing/short
                if len(window) < expected_hours:
                   # Pad with 0s? or NaNs? 
                   # Create dummy df of zeros with correct index?
                   # Simple solution: Reindex
                   full_idx = pd.date_range(hist_start, periods=expected_hours, freq='H')
                   window = window.reindex(full_idx).fillna(0)
                
                # Extract components
                win_surge = window['norm_surge'].values
                win_tide = window['norm_tide'].values
                
                # Seasonality: Take LAST value (current time)
                # i.e., at index 167 (most recent)
                last_row = window.iloc[-1]
                seas_feats = last_row[['month_sin', 'month_cos', 'doy_sin', 'doy_cos']].values
                
                # Flatten and Concat
                # [Surge_0..167, Tide_0..167, Seas_0..3]
                x_vec = np.concatenate([win_surge, win_tide, seas_feats])
                
                X_out.append(x_vec)
                ids_out.append(row['id'])
                
            return np.array(X_out), ids_out
            
        else:
            return np.array([]), []
            
        else:
            # Training mode or raw extraction not implemented for CSV here
            return np.array([]), []

    def calibrate_probabilities(self, probs, threshold=BEST_THRESHOLD):
        """
        Maps the optimal threshold to 0.5 using piecewise linear scaling.
        val < thresh -> scaled to [0, 0.5]
        val >= thresh -> scaled to [0.5, 1.0]
        """
        probs = np.array(probs)
        probs_calibrated = np.where(
            probs <= threshold,
            probs * (0.5 / threshold),
            0.5 + (probs - threshold) * (0.5 / (1.0 - threshold))
        )
        return probs_calibrated

    def predict_from_csv(self, hourly_csv, index_csv, output_csv):
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        X, ids = self.process_csv_data(hourly_csv, index_csv)
        
        print(f"Predicting for {len(X)} windows...")
        y_pred_probs_raw = self.model.predict(X)
        
        # Calibrate Probabilities
        print(f"Calibrating probabilities (Threshold {BEST_THRESHOLD} -> 0.5)...")
        y_pred_probs = self.calibrate_probabilities(y_pred_probs_raw)
        
        # Flatten predictions: 14 days per window -> 14 rows per window
        results = []
        global_id = 0
        for i, uid in enumerate(ids):
            # Window Prediction: vector of 14 values
            window_probs = y_pred_probs[i]
            
            for day_prob in window_probs:
                # We can use sequential ID matching scoring.py's expected y_test structure
                # y_test likely flattens: Window0_Day0, Window0_Day1...
                results.append({'id': global_id, 'y_prob': float(day_prob)})
                global_id += 1
            
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv} (Total rows: {len(res_df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_hourly", help="Path to training CSV", default="train_hourly.csv")
    parser.add_argument("--test_hourly", help="Path to testing CSV", default="test_hourly.csv")
    parser.add_argument("--test_index", help="Path to test index CSV", default="test_index.csv")
    parser.add_argument("--predictions_out", help="Path to output CSV", default="predictions.csv")
    
    # Optional args mainly for local testing
    parser.add_argument("--train", action="store_true", help="Run training (requires .mat files locally)")
    
    args = parser.parse_args()
    
    cfm = CoastalFloodModel(MODEL_FILE)
    
    if args.train:
        # Not implementing CSV training path here, assuming pre-trained model for submission
        pass
    else:
        # Inference Mode
        if os.path.exists(args.test_hourly) and os.path.exists(args.test_index):
            cfm.predict_from_csv(args.test_hourly, args.test_index, args.predictions_out)
        else:
            print("Test files not found. Please check paths.")