import scipy.io
import numpy as np
import os

DATA_FILE = r"data/NEUSTG_19502020_12stations.mat"
THRESH_FILE = r"data/Seed_Coastal_Stations_Thresholds.mat"

def analyze_constants():
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return

    data_mat = scipy.io.loadmat(DATA_FILE)
    thresh_mat = scipy.io.loadmat(THRESH_FILE)

    sea_level = data_mat['sltg']
    station_names = [s[0] for s in data_mat['sname'].flatten()]
    
    t_names = [s[0] for s in thresh_mat['sname'].flatten()]
    t_vals = thresh_mat['thminor_stnd'].flatten()
    threshold_map = dict(zip(t_names, t_vals))

    print("STATION_CONSTANTS = {")
    for i, name in enumerate(station_names):
        thresh = threshold_map.get(name)
        
        raw_hourly = sea_level[:, i]
        valid_data = raw_hourly[~np.isnan(raw_hourly)]
        std_dev = np.std(valid_data)
        
        print(f"    '{name}': {{'threshold': {thresh:.4f}, 'std': {std_dev:.4f}}},")
    print("}")

if __name__ == "__main__":
    analyze_constants()
