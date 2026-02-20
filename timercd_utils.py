import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class FloodDatasetStatic(Dataset):
    """Dataset with static covariates (Lat/Lon) for TimeRCD."""
    
    def __init__(self, data_path, geo_path, split='train', context_len=168, pred_len=336):
        """
        Args:
            data_path: Path to foundation_data.pkl
            geo_path: Path to station_thresholds_geo.csv with lat/lon info
            split: 'train' or 'test'
            context_len: Length of history window (Input)
            pred_len: Length of prediction window (Target)
        """
        import pandas as pd
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.station_data = data[split]
        self.context_len = context_len
        self.pred_len = pred_len
        self.full_len = context_len + pred_len
        
        # Load geo data
        geo_df = pd.read_csv(geo_path)
        self.geo_lookup = {}
        for _, row in geo_df.iterrows():
            self.geo_lookup[row['station_name']] = {
                'lat': row['lat'],
                'lon': row['lon']
            }
        
        # Normalize lat/lon to roughly same scale as sea level data
        # Lat: ~30-45 -> normalize to [-1, 1] range
        # Lon: ~-82 to -66 -> normalize to [-1, 1] range
        self.lat_mean = 38.0
        self.lat_std = 5.0
        self.lon_mean = -75.0
        self.lon_std = 5.0
        
        # Create an index mapping: global_idx -> (station_idx, sample_idx)
        self.index_map = []
        for s_idx, station in enumerate(self.station_data):
            num_samples = len(station['X'])
            for i in range(num_samples):
                self.index_map.append((s_idx, i))
                
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        s_idx, local_idx = self.index_map[idx]
        item = self.station_data[s_idx]
        
        station_name = item['name']
        
        # X: (168,) -> History
        # Y: (336,) -> Future
        X = item['X'][local_idx]  # (168,)
        Y = item['Y'][local_idx]  # (336,)
        
        # Concatenate to form the full sequence
        full_seq = np.concatenate([X, Y])  # (504,)
        
        # Get lat/lon for this station
        geo_info = self.geo_lookup.get(station_name, {'lat': self.lat_mean, 'lon': self.lon_mean})
        lat_norm = (geo_info['lat'] - self.lat_mean) / self.lat_std
        lon_norm = (geo_info['lon'] - self.lon_mean) / self.lon_std
        
        # Create static covariate channels (same value across time)
        lat_channel = np.full(self.full_len, lat_norm)  # (504,)
        lon_channel = np.full(self.full_len, lon_norm)  # (504,)
        
        # Stack: [Sea_Level, Lat, Lon] -> (504, 3)
        full_seq_3ch = np.stack([full_seq, lat_channel, lon_channel], axis=-1)  # (504, 3)
        full_seq_3ch = torch.FloatTensor(full_seq_3ch)
        
        # Mask for Reconstruction Task
        # 0 = Observed (History), 1 = Masked (Future)
        mask = torch.zeros(self.full_len, dtype=torch.bool)
        mask[self.context_len:] = True
        
        threshold = item['threshold']
        
        return {
            'time_series': full_seq_3ch,  # (504, 3)
            'mask': mask,                  # (504,)
            'threshold': threshold,
            'station_name': station_name
        }


class FloodDataset(Dataset):
    def __init__(self, data_path, split='train', context_len=168, pred_len=336, augment=False):
        """
        Args:
            data_path: Path to foundation_data.pkl
            split: 'train' or 'test'
            context_len: Length of history window (Input)
            pred_len: Length of prediction window (Target)
            augment: Boolean, whether to apply data augmentation (scaling, jitter)
        """
        self.augment = augment
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.station_data = data[split] # List of dicts
        self.context_len = context_len
        self.pred_len = pred_len
        self.full_len = context_len + pred_len
        
        # Create an index mapping: global_idx -> (station_idx, sample_idx)
        self.index_map = []
        for s_idx, station in enumerate(self.station_data):
            num_samples = len(station['X'])
            for i in range(num_samples):
                self.index_map.append((s_idx, i))
                
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        s_idx, local_idx = self.index_map[idx]
        item = self.station_data[s_idx]
        
        # X: (N,) -> History (may be longer than context_len in deep datasets)
        # Y: (336,) -> Future
        X = item['X'][local_idx]
        Y = item['Y'][local_idx]
        
        # Dynamic slicing: take the LAST context_len hours from X
        if len(X) > self.context_len:
            X = X[-self.context_len:]
        
        full_seq = np.concatenate([X, Y])
        
        # Apply Data Augmentation (only if enabled)
        if self.augment:
            # 1. Random Scaling (0.9 to 1.1)
            scale = np.random.uniform(0.9, 1.1)
            full_seq = full_seq * scale
            
            # 2. Jitter (Gaussian noise, sigma=0.01)
            noise = np.random.normal(0, 0.01, size=full_seq.shape)
            full_seq = full_seq + noise
            
        full_seq = torch.FloatTensor(full_seq).unsqueeze(-1)  # (context_len + pred_len, 1)
        
        # Mask for Reconstruction Task
        # 0 = Observed (History), 1 = Masked (Future)
        mask = torch.zeros(self.full_len, dtype=torch.bool)
        mask[self.context_len:] = True # Mask the prediction part
        
        threshold = item['threshold']
        
        # Flood label: did ANY hour in the future exceed threshold (normalized value > 0)?
        flood_label = int((Y > 0).any())
        
        return {
            'time_series': full_seq,   # (504, 1)
            'mask': mask,              # (504,)
            'threshold': threshold,
            'station_name': item['name'],
            'flood_label': flood_label  # NEW: for FloodScout training
        }

