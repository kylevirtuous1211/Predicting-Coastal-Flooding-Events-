import pickle
import numpy as np
import os

DATA_FILE = "foundation_data_deep.pkl"
try:
    if not os.path.exists(DATA_FILE):
        print(f"File {DATA_FILE} not found. Trying foundation_data.pkl...")
        DATA_FILE = "foundation_data.pkl"

    print(f"Loading {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print("Keys:", data.keys())
    
    if 'test' in data:
        test_data = data['test']
        print(f"Number of test stations: {len(test_data)}")
        
        total_samples = 0
        flood_samples = 0
        non_flood_samples = 0
        
        for s_idx, station in enumerate(test_data):
            # Check Y length. Usually Y is the future sequence.
            # We want to check next 14 days (336 hours).
            # Assuming Y contains at least 336 hours.
            
            Ys = station['Y']
            num_station_samples = len(Ys)
            print(f"  Station {s_idx}: {station.get('name', 'Unknown')} - {num_station_samples} samples")
            
            for y_seq in Ys:
                # y_seq shape: (pred_len,) or (pred_len, 1)
                # We care about the first 336 hours if it's longer
                target_seq = y_seq[:336]
                
                is_flood = (target_seq > 0).any()
                
                if is_flood:
                    flood_samples += 1
                else:
                    non_flood_samples += 1
            
            total_samples += num_station_samples

        print("\n=== Test Data Distribution (Next 14 Days > 0) ===")
        print(f"Total Samples: {total_samples}")
        print(f"Flood Samples (1): {flood_samples} ({flood_samples/total_samples:.2%})")
        print(f"Non-Flood Samples (0): {non_flood_samples} ({non_flood_samples/total_samples:.2%})")
            
except Exception as e:
    print(f"Error: {e}")
