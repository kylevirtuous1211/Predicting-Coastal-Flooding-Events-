#!/bin/bash

# Prep Directory
mkdir -p submission_prior
rm -rf submission_prior/*

# Copy Files
cp model_prior_token.py submission_prior/model.py
cp requirements.txt submission_prior/Requirements.txt
cp README.md submission_prior/
cp station_metadata.pkl submission_prior/

# Copy Checkpoints
cp checkpoints/scout/scout_best.pth submission_prior/scout.pkl
cp checkpoints/timercd_prior_FiLM/timercd_prior_best_contextlen=1800.pth submission_prior/model.pkl

# Zip
cd submission_prior
zip -r ../submission_prior.zip .
cd ..

echo "Submission file created: submission_prior.zip"
