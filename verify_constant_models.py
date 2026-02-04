import pandas as pd
import subprocess
import os
import sys

def verify_model(script_name, expected_val):
    print(f"Verifying {script_name} (Expected: {expected_val})...")
    
    # Create dummy test files
    os.makedirs("tmp_verify", exist_ok=True)
    test_index = pd.DataFrame({'id': [0, 1, 2, 3, 4], 'station_name': ['A']*5, 'hist_start':['2020-01-01']*5, 'hist_end':['2020-01-01']*5, 'future_start':['2020-01-01']*5, 'future_end':['2020-01-01']*5})
    test_index.to_csv("tmp_verify/test_index.csv", index=False)
    
    cmd = [
        sys.executable, script_name,
        "--test_index", "tmp_verify/test_index.csv",
        "--predictions_out", "tmp_verify/predictions.csv"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: Script crashed.\n{result.stderr}")
        return False
        
    if not os.path.exists("tmp_verify/predictions.csv"):
        print("FAILED: No predictions.csv created.")
        return False
        
    preds = pd.read_csv("tmp_verify/predictions.csv")
    if 'label' not in preds.columns:
        print("FAILED: 'label' column missing.")
        return False
        
    vals = preds['label'].unique()
    if len(vals) == 1 and vals[0] == expected_val:
        print("SUCCESS: Output matches expectation.")
        return True
    else:
        print(f"FAILED: Unexpected values: {vals}")
        return False

print("--- Test Start ---")
v1 = verify_model("model_ones.py", 1)
v2 = verify_model("model_zeros.py", 0)

if v1 and v2:
    print("\nALL TESTS PASSED")
else:
    print("\nXXX TESTS FAILED XXX")
