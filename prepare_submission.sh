#!/bin/bash

# Prep Directories
mkdir -p init_submission
mkdir -p init_submission/zeroshot
mkdir -p init_submission/finetuned
mkdir -p init_submission/geo
mkdir -p init_submission/geo/analysis_plots
mkdir -p init_submission/prior

# Common Files
cp model.py init_submission/zeroshot/
cp requirements.txt init_submission/zeroshot/Requirements.txt
cp README.md init_submission/zeroshot/
cp station_metadata.pkl init_submission/zeroshot/

cp model.py init_submission/finetuned/
cp requirements.txt init_submission/finetuned/Requirements.txt
cp README.md init_submission/finetuned/
cp station_metadata.pkl init_submission/finetuned/

# Zero-Shot Model
cp Time-RCD/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth init_submission/zeroshot/model.pkl

# Finetuned Model
cp checkpoints/timercd_finetune/timercd_epoch_19.pth init_submission/finetuned/model.pkl

# Geo Model (best: epoch 2)
cp checkpoints/timercd_geo/geo_epoch_2.pth init_submission/geo/model.pkl

# Prior Model (epoch 4)
cp model_prior_token.py init_submission/prior/model.py
cp requirements.txt init_submission/prior/Requirements.txt
cp README.md init_submission/prior/
cp station_metadata.pkl init_submission/prior/
cp checkpoints/scout/scout_best.pth init_submission/prior/scout.pkl
cp checkpoints/timercd_prior/timercd_prior_epoch_4.pth init_submission/prior/model.pkl

# Zip
rm -f zeroshot.zip finetuned.zip geo.zip prior.zip
cd init_submission/zeroshot
# zip -r ../../zeroshot.zip .
# cd ../finetuned
# zip -r ../../finetuned.zip .
cd ../prior
zip -r ../../prior.zip .

echo "Submission files created: zeroshot.zip, finetuned.zip, and prior.zip"
