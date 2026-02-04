import argparse
import pandas as pd
import sys

def predict(args):
    print("Running Constant Ones Model")
    # Read the test index to get IDs
    try:
        test_index = pd.read_csv(args.test_index)
    except Exception as e:
        print(f"Error reading test index: {e}")
        sys.exit(1)
    
    # Create predictions
    ids = test_index['id']
    # Constant 1
    labels = [1] * len(ids)
    
    results = pd.DataFrame({'id': ids, 'label': labels})
    results.to_csv(args.predictions_out, index=False)
    print(f"Predictions saved to {args.predictions_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept all arguments ingestion passes
    parser.add_argument("--train_hourly", type=str)
    parser.add_argument("--test_hourly", type=str)
    parser.add_argument("--test_index", type=str)
    parser.add_argument("--predictions_out", type=str)
    
    # Extra args to prevent crashing if extra flags are passed
    parser.add_argument("--mode", type=str, default="evaluate") 
    parser.add_argument("--data", type=str, default="foundation_data.pkl")
    parser.add_argument("--model", type=str, default="model.pkl")

    args, unknown = parser.parse_known_args()
    
    if args.test_index and args.predictions_out:
        predict(args)
    else:
        print("Usage: model.py --test_index <path> --predictions_out <path>")
