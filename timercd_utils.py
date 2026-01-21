import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class FloodDataset(Dataset):
    def __init__(self, data_path, split='train', context_len=168, pred_len=336):
        """
        Args:
            data_path: Path to foundation_data.pkl
            split: 'train' or 'test'
            context_len: Length of history window (Input)
            pred_len: Length of prediction window (Target)
        """
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
        
        # X: (168,) -> History
        # Y: (336,) -> Future
        X = item['X'][local_idx] # (168,)
        Y = item['Y'][local_idx] # (336,)
        
        # Concatenate to form the full sequence for TimeRCD
        full_seq = np.concatenate([X, Y]) # (504,)
        full_seq = torch.FloatTensor(full_seq).unsqueeze(-1) # (504, 1)
        
        # Mask for Reconstruction Task
        # 0 = Observed (History), 1 = Masked (Future)
        mask = torch.zeros(self.full_len, dtype=torch.bool)
        mask[self.context_len:] = True # Mask the prediction part
        
        threshold = item['threshold']
        
        return {
            'time_series': full_seq,   # (504, 1)
            'mask': mask,              # (504,)
            'threshold': threshold,
            'station_name': item['name']
        }
