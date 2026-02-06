#!/bin/bash
# Prepare TimeRCD Prior Token Submission
# Creates a zip file ready for Codabench upload

set -e

SUBMISSION_DIR="submission_prior"
ZIP_NAME="prior_submission.zip"

# Allow specifying checkpoint via argument, default to best
CHECKPOINT="${1:-checkpoints/timercd_prior/timercd_prior_best.pth}"

echo "Preparing TimeRCD Prior Token submission..."
echo "Using checkpoint: $CHECKPOINT"

# Clean up previous submission
rm -rf $SUBMISSION_DIR
mkdir -p $SUBMISSION_DIR

# Copy model file (rename to model.py for ingestion)
cp model_prior_token.py $SUBMISSION_DIR/model.py

# Copy checkpoint (rename to model.pkl - the model expects this)
cp $CHECKPOINT $SUBMISSION_DIR/model.pkl

# Copy scout weights (still needed for FloodScout in the prior model)
cp scout.pkl $SUBMISSION_DIR/scout.pkl

# Copy metadata for proper normalization
cp station_metadata.pkl $SUBMISSION_DIR/station_metadata.pkl

# List contents
echo "Submission contents:"
ls -la $SUBMISSION_DIR/

# Create zip
cd $SUBMISSION_DIR
zip -r ../$ZIP_NAME .
cd ..

echo ""
echo "============================================"
echo "✅ Prior Token Submission created: $ZIP_NAME"
echo "============================================"
echo ""
echo "Contents:"
unzip -l $ZIP_NAME
echo ""
echo "Upload this file to Codabench!"
echo ""
echo "To use a different checkpoint:"
echo "  ./prepare_scout_submission.sh checkpoints/timercd_prior/timercd_prior_epoch_N.pth"
