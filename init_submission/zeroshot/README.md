# TimeRCD for Coastal Flooding Prediction

## Overview
This repository contains the implementation of TimeRCD, a foundation model for time series forecasting, applied to the prediction of coastal flooding events. The model takes 7 days of historical sea level data (hourly) and predicts the next 14 days.

## File Structure
- `model.py`: The main script containing the model architecture (TimeRCD), data loading, and evaluation logic.
- `model.pkl`: The trained model weights.
- `Requirements.txt`: List of dependencies.

## Installation
Install the required libraries:
```bash
pip install -r Requirements.txt
```

## Usage

### Evaluation
 To evaluate the model using the provided `model.pkl` and `foundation_data.pkl`:
```bash
python model.py --mode evaluate --data foundation_data.pkl --model model.pkl
```
This will output the Confusion Matrix and Matthews Correlation Coefficient (MCC).

### Training/Finetuning
To finetune the model (if data is available):
```bash
python model.py --mode train --data foundation_data.pkl --model model.pkl
```

## Model Details
- **Architecture**: TimeRCD (Time-Series Rotary-position-embedding Cross-domain)
- **Input**: 168 hours (7 days) of sea level data.
- **Output**: 336 hours (14 days) of future sea level data.
- **Normalization**: Data is normalized per station using `(value - threshold) / std`, effectively setting the flood threshold to 0.0.
- **Prediction**: The model predicts the future sequence. Any hour > 0.0 in the predicted window indicates a flood event.

## Results
- **Zero-Shot Score (MCC)**: 0.8941
- **Finetuned Score (MCC)**: 0.8866 (Optimized for Recall)
