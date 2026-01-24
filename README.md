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
- `model.py`: The core script for the challenge submission. Contains the model architecture, inference logic, and evaluation loop.
- `train_timercd.py`: Designated script for training the TimeRCD model from scratch or finetuning.
- `preprocess_foundation.py`: Script to prepare and normalize data from raw sources into `foundation_data.pkl`.
- `timercd_utils.py`: Utility functions and Dataset classes shared across training scripts.
- `extract_metadata.py`: Helper to extract station statistics (mean, std, thresholds) for normalization.

## Usage

### 1. Data Preprocessing
If starting from raw data:
```bash
python preprocess_foundation.py
```
This generates `foundation_data.pkl` containing training and testing splits.

### 2. Training
To train the model:
```bash
python train_timercd.py
```
Checkpoints will be saved to `checkpoints/timercd_finetune/`.

### 3. Evaluation & Submission
To run the model in evaluation mode (as used by the competition platform):
```bash
python model.py --mode evaluate --data foundation_data.pkl --model model.pkl
```

For submission capability, `model.py` is self-contained. It loads `model.pkl` (weights) and `station_metadata.pkl` (normalization stats).

## Model Architecture details
- **Backbone**: TimeRCD (Time-Series Rotary-position-embedding Cross-domain)
- **Input Dimensions**: 168 time steps (7 days) x 1 feature (Sea Level)
- **Output**: 336 time steps (14 days) reconstruction
- **Flood Detection**:
  - The model outputs normalized sea level predictions.
  - Flooding is predicted if any hourly value in the future window exceeds the station's flood threshold (normalized to 0.0).

## Results
- **Metric**: Matthews Correlation Coefficient (MCC) & F1 Score.
- **Current Performance**: Zero-shot MCC ~0.89.
