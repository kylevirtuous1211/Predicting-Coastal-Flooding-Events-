"""
1D-CNN FloodScout Model for Coastal Flooding Prediction

This model uses a simple 1D Convolutional Neural Network to predict
flood events from 7-day (168-hour) history data.

For Codabench submission: rename this file to model.py

Usage:
    python model.py \
        --train_hourly <train_csv> \
        --test_hourly <test_csv> \
        --test_index <index_csv> \
        --predictions_out <output_csv>
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
import argparse
from tqdm import tqdm


# ==========================================
# 1. FloodScout Model (1D-CNN)
# ==========================================

class FloodScout(nn.Module):
    """
    Lightweight 1D-CNN that predicts flood probability from 7-day history.
    
    Input: (B, 168, 1) - 7-day hourly sea level history
    Output: (B, 1) - flood probability in [0, 1]
    """
    def __init__(self, input_len: int = 168, hidden_dim: int = 64):
        super().__init__()
        self.input_len = input_len
        
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 168 -> 84
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 84 -> 42
            
            # Third conv block
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # -> 1
            
            nn.Flatten(),  # (B, hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 168, 1) - history sequence
        Returns:
            (B, 1) - flood probability
        """
        # Reshape for Conv1d: (B, 168, 1) -> (B, 1, 168)
        x = x.permute(0, 2, 1)
        features = self.cnn(x)
        prob = self.classifier(features)
        return prob


# ==========================================
# 2. Ingestion Dataset (168-hour history ONLY)
# ==========================================

class IngestionDataset(Dataset):
    """
    Dataset for ingestion that loads ONLY 168-hour history.
    No future data is loaded - this prevents any data leakage.
    """
    def __init__(self, train_csv, test_csv, test_index_csv, metadata_path=None, context_len=168):
        self.context_len = context_len
        
        # Load metadata if available (for proper normalization)
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded metadata for {len(self.metadata)} stations")
        
        print("Loading CSVs...")
        train_df = pd.read_csv(train_csv, parse_dates=['time'])
        test_df = pd.read_csv(test_csv, parse_dates=['time'])
        self.test_index = pd.read_csv(test_index_csv)
        
        # Combine train and test for continuous history
        self.full_df = pd.concat([train_df, test_df], ignore_index=True)
        self.full_df = self.full_df.sort_values(['station_name', 'time']).reset_index(drop=True)
        
        # Preprocess station data
        self.station_data = {}
        for name, group in self.full_df.groupby('station_name'):
            group = group.set_index('time').sort_index()
            vals = group['sea_level'].values
            
            # Interpolate missing values
            vals = pd.Series(vals).interpolate(limit=24).fillna(method='bfill').fillna(method='ffill').values
            
            # Normalize based on metadata or estimate
            meta = self.metadata.get(name)
            if meta:
                thresh = meta['threshold']
                std = meta['std']
                norm_vals = (vals - thresh) / std
            else:
                # OOD station: estimate threshold
                std = np.std(vals)
                if std < 1e-6:
                    std = 1.0
                estimated_thresh = np.mean(vals) + 2.29 * std
                norm_vals = (vals - estimated_thresh) / std
                print(f"OOD station {name}: estimated threshold = {estimated_thresh:.4f}")
            
            self.station_data[name] = {
                'values': norm_vals,
                'times': group.index
            }
        
        print(f"Loaded {len(self.test_index)} test samples")
    
    def __len__(self):
        return len(self.test_index)
    
    def __getitem__(self, idx):
        row = self.test_index.iloc[idx]
        station_name = row['station_name']
        hist_start = pd.to_datetime(row['hist_start'])
        
        data_info = self.station_data.get(station_name)
        if data_info is None:
            print(f"Warning: Station {station_name} not found!")
            # Return zeros as fallback
            return {
                'history': torch.zeros(self.context_len, 1),
                'id': row['id'],
                'station_name': station_name
            }
        
        times = data_info['times']
        values = data_info['values']
        
        # Find the starting index for history
        start_idx = times.searchsorted(hist_start)
        
        # Extract ONLY the 168-hour history - NO FUTURE
        if start_idx + self.context_len > len(values):
            # Handle edge case: pad if not enough data
            hist_seq = values[start_idx:]
            pad_len = self.context_len - len(hist_seq)
            hist_seq = np.pad(hist_seq, (0, pad_len), mode='edge')
        else:
            hist_seq = values[start_idx:start_idx + self.context_len]
        
        # Convert to tensor: (168,) -> (168, 1)
        history = torch.FloatTensor(hist_seq).unsqueeze(-1)
        
        return {
            'history': history,  # (168, 1) - ONLY HISTORY
            'id': row['id'],
            'station_name': station_name
        }


# ==========================================
# 3. Ingestion Prediction Function
# ==========================================

def ingestion_predict(args):
    """Main prediction function for Codabench ingestion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    metadata_path = "station_metadata.pkl"
    checkpoint_path = "scout.pkl"  # FloodScout trained weights
    
    # Optimal threshold from training (calibrated for best MCC)
    THRESHOLD = 0.45
    
    # Load dataset
    dataset = IngestionDataset(
        args.train_hourly,
        args.test_hourly,
        args.test_index,
        metadata_path=metadata_path if os.path.exists(metadata_path) else None,
        context_len=168
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    print("Initializing FloodScout...")
    model = FloodScout(input_len=168).to(device)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            if 'threshold' in state_dict:
                THRESHOLD = state_dict.get('threshold', THRESHOLD)
                print(f"Using threshold from checkpoint: {THRESHOLD}")
        
        # Handle 'module.' prefix from DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        print("Model weights loaded successfully!")
    else:
        print(f"ERROR: Checkpoint {checkpoint_path} not found!")
        return
    
    model.eval()
    results = []
    
    print(f"Predicting with threshold = {THRESHOLD}")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            history = batch['history'].to(device)  # (B, 168, 1)
            ids = batch['id']
            
            # Forward pass - direct classification
            probs = model(history)  # (B, 1)
            
            # Apply threshold
            for i in range(len(ids)):
                prob = probs[i].item()
                label = 1 if prob > THRESHOLD else 0
                results.append({
                    'id': ids[i].item() if torch.is_tensor(ids[i]) else ids[i],
                    'label': label
                })
    
    # Save predictions
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.predictions_out, index=False)
    print(f"Predictions saved to {args.predictions_out}")
    
    # Print summary
    num_flood = sum(1 for r in results if r['label'] == 1)
    print(f"Summary: {num_flood}/{len(results)} predicted as FLOOD ({100*num_flood/len(results):.1f}%)")


# ==========================================
# 4. Main Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D-CNN FloodScout Prediction")
    
    # Ingestion arguments (required by Codabench)
    parser.add_argument("--train_hourly", type=str, required=True,
                        help="Path to train_hourly.csv")
    parser.add_argument("--test_hourly", type=str, required=True,
                        help="Path to test_hourly.csv")
    parser.add_argument("--test_index", type=str, required=True,
                        help="Path to test_index.csv")
    parser.add_argument("--predictions_out", type=str, required=True,
                        help="Path to output predictions.csv")
    
    args = parser.parse_args()
    ingestion_predict(args)
