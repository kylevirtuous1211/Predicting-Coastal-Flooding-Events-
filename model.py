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
        print(f"Reading {hourly_csv_path}...")
        df = pd.read_csv(hourly_csv_path)
        
        # Ensure time is datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # 1. Normalize and Aggregate to Daily
        print("Aggregating to daily stats...")
        daily_records = []
        
        # We process each station separately to be efficient
        for station_name, group in df.groupby('station_name'):
            # Get constants
            consts = STATION_CONSTANTS.get(station_name)
            if not consts:
                continue
            
            thresh = consts['threshold']
            std_dev = consts['std']
            
            # Calculate Normalized Value
            # (SL - Thresh) / Std
            group = group.copy()
            group['norm_val'] = (group['sea_level'] - thresh) / std_dev
            
            # Resample to Daily
            # We want to keep station_name
            g_daily = group.set_index('time').resample('D').agg({
                'sea_level': 'max',
                'norm_val': ['mean', 'max', 'min', 'std']
            })
            
            # Flatten columns
            g_daily.columns = ['_'.join(col).strip() for col in g_daily.columns.values]
            g_daily = g_daily.rename(columns={
                'sea_level_max': 'max_raw_sl',
                'norm_val_mean': 'feat_mean',
                'norm_val_max':  'feat_max',
                'norm_val_min':  'feat_min',
                'norm_val_std':  'feat_std'
            })
            
            g_daily['station_name'] = station_name
            daily_records.append(g_daily)
            
        if not daily_records:
            return np.array([]), []
            
        df_daily_all = pd.concat(daily_records)
        df_daily_all = df_daily_all.sort_index() # Sort by time
        
        # 2. Extract Windows based on Index
        if index_csv_path:
            print(f"Extracting windows from {index_csv_path}...")
            index_df = pd.read_csv(index_csv_path)
            
            X_out = []
            ids_out = []
            
            # Optimize: Group Daily by Station for fast lookup
            daily_by_station = {k: v for k, v in df_daily_all.groupby('station_name')}
            
            for _, row in index_df.iterrows():
                stn = row['station_name']
                hist_start = pd.to_datetime(row['hist_start'])
                hist_end = pd.to_datetime(row['hist_end'])
                row_id = row['id']
                
                if stn not in daily_by_station:
                    continue
                
                stn_data = daily_by_station[stn]
                
                # Slicing: inclusive of start and end for daily data?
                # hist_start to hist_end is 7 days.
                # Check timestamps. 
                # pandas slicing on DatetimeIndex is inclusive.
                mask = (stn_data.index >= hist_start) & (stn_data.index <= hist_end)
                window = stn_data[mask]
                
                # Feature columns
                feat_cols = ['feat_mean', 'feat_max', 'feat_min', 'feat_std']
                
                if len(window) != HIST_WINDOW:
                    # Pad? Or Skip? 
                    # Challenge data should be complete, but robust code checks.
                    # If missing, we might pad with 0 or last value using reindex
                    expected_dates = pd.date_range(start=hist_start, end=hist_end, freq='D')
                    window = window.reindex(expected_dates).fillna(0) # Simple imputation
                
                assert len(window) == HIST_WINDOW
                
                feats = window[feat_cols].values.flatten()
                X_out.append(feats)
                ids_out.append(row_id)
                
            return np.array(X_out), ids_out
            
        else:
            # Training mode or raw extraction not implemented for CSV here
            return np.array([]), []

    def predict_from_csv(self, hourly_csv, index_csv, output_csv):
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        X, ids = self.process_csv_data(hourly_csv, index_csv)
        
        print(f"Predicting for {len(X)} windows...")
        y_pred_prob = self.model.predict(X)
        # Binary prediction (Optional: Return probabilities? Challenge asks for binary usually, but ingestion checks 'y_prob')
        # We will return the probability/label. Since ingestion checks for 'y_prob', let's format it.
        # Actually XGBRegressor predicts continuous score. Since we trained with binary targets 0/1, 
        # the output is probability-like.
        
        # Formatting output
        results = []
        for i, uid in enumerate(ids):
            # Convert 14-day array to list string or similar?
            # If the ingestion expects a single column 'y_prob', it's likely a list string.
            # OR we output 14 columns?
            # Let's try outputting the list as a string to be safe.
            preds = y_pred_prob[i].tolist()
            results.append({'id': uid, 'y_prob': str(preds)})
            
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

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