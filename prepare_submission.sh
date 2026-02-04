#!/bin/bash

# Prep Directories
mkdir -p init_submission
mkdir -p init_submission/zeroshot
mkdir -p init_submission/finetuned
mkdir -p init_submission/geo
mkdir -p init_submission/geo/analysis_plots

# Common Files
cp model.py init_submission/zeroshot/
cp requirements.txt init_submission/zeroshot/Requirements.txt
cp README.md init_submission/zeroshot/
cp station_metadata.pkl init_submission/zeroshot/

cp model.py init_submission/finetuned/
cp requirements.txt init_submission/finetuned/Requirements.txt
cp README.md init_submission/finetuned/
cp station_metadata.pkl init_submission/finetuned/

# Geo Submission (with lat/lon covariates)
cp model.py init_submission/geo/
cp requirements.txt init_submission/geo/Requirements.txt
cp README.md init_submission/geo/
cp station_metadata.pkl init_submission/geo/
cp analysis_plots/station_thresholds_geo.csv init_submission/geo/analysis_plots/

# Zero-Shot Model
cp Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth init_submission/zeroshot/model.pkl

# Finetuned Model
cp checkpoints/timercd_finetune/timercd_epoch_1.pth init_submission/finetuned/model.pkl

# Geo Model (best: epoch 2)
cp checkpoints/timercd_geo/geo_epoch_2.pth init_submission/geo/model.pkl

# Zip
rm -f zeroshot.zip finetuned.zip geo.zip
cd init_submission/zeroshot
zip -r ../../zeroshot.zip .
cd ../finetuned
zip -r ../../finetuned.zip .
cd ../geo
zip -r ../../geo.zip .

echo "Submission files created: zeroshot.zip, finetuned.zip, and geo.zip"
