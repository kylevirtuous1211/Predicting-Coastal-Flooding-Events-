# TimeRCD for Coastal Flooding Prediction

## Overview
This repository implements **TimeRCD**, a foundation model for time series forecasting, tailored for predicting coastal flooding events. The model utilizes a Transformer-based architecture with Rotary Positional Embeddings (RoPE) to forecast sea levels 14 days into the future based on 7 days of historical hourly data.

## Pipeline Steps
1.  **Data Preprocessing**: Raw station data is processed, normalized, and formatted for the model.
2.  **Training**: The TimeRCD model is trained (or finetuned) on the processed dataset.
3.  **Inference/Evaluation**: The model generates predictions which are thresholded to determine flooding events (Binary Classification).

## Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Key Files
- `model_timercd.py`: Core model script for the challenge. Contains architecture and ingestion logic.
- `train_timercd.py`: Main script for finetuning TimeRCD on coastal flooding data.
- `preprocess_foundation_deep.py`: Generates the deep context dataset (`foundation_data_deep_105d.pkl`).
- `sweep_timercd.py`: Performs a threshold sweep on checkpoints to optimize for MCC.
- `visualize_predictions.py`: Generates plots for the 15 official seed intervals.

## Reproduction Steps

### 1. Data Generation
Generate the 105-day context dataset (75 days history + 14 days forecasting window + 16 days padding/slack):
```bash
python preprocess_foundation_deep.py --hist_days 105
```
This produces `foundation_data_deep_105d.pkl`.

### 2. Fine-tuning
Ensure the hyperparameters in `train_timercd.py` match the best configuration:
*   `CONTEXT_LEN = 1800` (75 days)
*   `PATCH_SIZE = 21`
*   `FLOOD_WEIGHT = 8.0`

Run the training script:
```bash
python train_timercd.py
```
Checkpoints are saved in `checkpoints/timercd_finetune/75days/`.

### 3. Threshold Optimization
Run a sweep on the best checkpoint (e.g., epoch 60) to find the optimal decision boundary:
```bash
python sweep_timercd.py --checkpoint checkpoints/timercd_finetune/75days/timercd_epoch_60.pth
```
**Best Result**: MCC ≈ 0.6289 at Threshold = **-0.10**.

### 4. Visualization & Verification
Generate plots for the official seed intervals to verify phase alignment and reconstruction quality:
```bash
# Update CHECKPOINT in visualize_predictions.py if necessary
python visualize_predictions.py
```

## Configuration & Architecture
The model is optimized for **long-term tidal memory** and **imbalanced flood events**.

- **Context Length**: 1800 hours (75 days). High-resolution context allows the Transformer to align with local station tidal phases.
- **Patch Size**: 21 hours. Optimized for sea-level frequency.
- **Loss Weighting**: 8.0x penalty for flood events (Values > Threshold).
- **Thresholding**: Data is normalized as `(Value - Threshold) / Std`, meaning a prediction > 0 theoretically signifies a flood. The optimal threshold found via sweep is **-0.10**.

## Performance
- **Zero-shot (Pretrained)**: MCC ~0.33
- **Fine-tuned (75d context)**: **MCC 0.6289**, **F1 0.6655** (at Epoch 60).
