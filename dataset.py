import pickle
import os

class FloodDataset:
    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = "processed_data.pkl"
        else:
            self.data_path = data_path
            
        self.data = self._load()

    def _load(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed data not found at {self.data_path}. Run preprocess_data.py first.")
        
        print(f"Loading dataset from {self.data_path}...")
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_train_data(self):
        """Returns X_train, y_train"""
        return self.data['X_train'], self.data['y_train']

    def get_val_data(self):
        """Returns X_val, y_val"""
        return self.data['X_val'], self.data['y_val']
    
    def get_feature_names(self):
        return self.data.get('feature_names', [])

if __name__ == "__main__":
    # Test loading
    ds = FloodDataset()
    xt, yt = ds.get_train_data()
    print(f"Loaded Train: {xt.shape}, {yt.shape}")
