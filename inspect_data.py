
import pickle
import numpy as np

DATA_FILE = "foundation_data.pkl"
try:
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print("Keys:", data.keys())
    if 'train' in data:
        station0 = data['train'][0]
        print("Station 0 keys:", station0.keys())
        print("Number of samples:", len(station0['X']))
        if len(station0['X']) > 0:
            print("X[0] shape:", getattr(station0['X'][0], 'shape', len(station0['X'][0])))
            print("Y[0] shape:", getattr(station0['Y'][0], 'shape', len(station0['Y'][0])))
            
            # Check if it's a list sample
            print("Type of X:", type(station0['X']))
            print("Type of X[0]:", type(station0['X'][0]))
            
except Exception as e:
    print(f"Error: {e}")
