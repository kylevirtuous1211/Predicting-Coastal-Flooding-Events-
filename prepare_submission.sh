#!/bin/bash

# Prep Directories
mkdir -p init_submission
mkdir -p init_submission/zeroshot
mkdir -p init_submission/finetuned

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
cp checkpoints/timercd_finetune/timercd_epoch_1.pth init_submission/finetuned/model.pkl

# Zip
rm -f zeroshot.zip finetuned.zip
cd init_submission/zeroshot
zip -r ../../zeroshot.zip .
cd ../finetuned
zip -r ../../finetuned.zip .

echo "Submission files created: zeroshot.zip and finetuned.zip"
