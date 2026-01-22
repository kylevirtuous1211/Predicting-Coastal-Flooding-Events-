import pickle
import numpy as np

with open('foundation_data.pkl', 'rb') as f:
    data = pickle.load(f)

metadata = {}
# Process Train stats
for item in data['train']:
    metadata[item['name']] = {
        'threshold': item['original_threshold'],
        'mean': item['mean'],
        'std': item['std']
    }

# Process Test stats
for item in data['test']:
    metadata[item['name']] = {
        'threshold': item['original_threshold'],
        'mean': item['mean'],
        'std': item['std']
    }
    
print(f"Extracted metadata for {len(metadata)} stations.")

with open('station_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
