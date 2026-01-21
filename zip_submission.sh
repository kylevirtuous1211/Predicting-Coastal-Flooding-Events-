#!/bin/bash

# Define output filename
ZIP_NAME="optimize_mcc.zip"

# Copy latest model checkpoint to root
if [ -f "checkpoints/xgb_global_model.pkl" ]; then
    echo "Copying checkpoint to model.pkl..."
    cp checkpoints/xgb_global_model.pkl model.pkl
else
    echo "Warning: checkpoints/xgb_global_model.pkl not found. using existing model.pkl if available."
fi

# Remove old zip if exists
if [ -f "$ZIP_NAME" ]; then
    rm "$ZIP_NAME"
fi

# Zip files
# Check if zip command exists
if ! command -v zip &> /dev/null; then
    echo "Error: 'zip' command not found. Please install zip or use the Python one-liner."
    exit 1
fi

echo "Zipping files into $ZIP_NAME..."
zip "$ZIP_NAME" model.py model.pkl requirements.txt README.md

echo "Done: $ZIP_NAME created."
