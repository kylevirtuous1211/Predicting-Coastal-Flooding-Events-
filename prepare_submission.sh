#!/bin/bash

# Prep Directory
mkdir -p submission_timercd
rm -rf submission_timercd/*
rm -f submission_timercd.zip

# Copy Files
cp model_timercd.py submission_timercd/model.py
cp requirements.txt submission_timercd/Requirements.txt
cp README.md submission_timercd/
cp station_metadata.pkl submission_timercd/

# Copy Checkpoints
cp checkpoints/timercd_finetune/75days/timercd_epoch_60.pth submission_timercd/model.pkl

# Zip
cd submission_timercd
zip -r ../submission_timercd.zip .
cd ..

echo "Submission file created: submission_timercd.zip"
