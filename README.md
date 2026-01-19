# Predicting Coastal Flooding Events - Model Submission

## Overview
This submission contains a Global XGBoost model designed to predict coastal flooding events 14 days into the future based on 7 days of historical sea level data.

## Pipeline Description

### 1. Preprocessing (`model.py`)
The model uses a **Standardized Distance to Threshold** strategy to handle Out-of-Distribution (OOD) stations.
*   **Normalization:** For each station, we calculate `(Sea_Level - Threshold) / Station_Std_Dev`.
*   **Feature Engineering:** We aggregate hourly data into Daily Statistics:
    *   Mean, Max, Min, and Std Dev of the standardized values.
*   **Input Window:** 7 Days (7 * 4 = 28 features).
*   **Output:** Binary classification (Flood/No-Flood) for the next 14 days.

### 2. Model Architecture
*   **Algorithm:** XGBoost (Gradient Boosted Trees).
*   **Wrapper:** `MultiOutputRegressor` (trains 14 separate regressors, one for each forecast day).
*   **Hyperparameters:**
    *   `n_estimators`: 100
    *   `max_depth`: 6
    *   `learning_rate`: 0.1
    *   `objective`: binary:logistic

### 3. Training & Evaluation
The model was training on 9 stations and validated on 3 held-out stations (Lewes, Fernandina Beach, The Battery).
*   **Accuracy:** ~98.8%
*   **MCC:** ~0.10

## reproducibility

### Prerequisites
Install dependencies:
```bash
pip install -r requirements.txt
```

### Files
*   `model.py`: Contains the `CoastalFloodModel` class with full preprocessing, training, and evaluation logic.
*   `model.pkl`: The trained model weights.
*   `requirements.txt`: Python dependencies.

### Running the Code
To evaluate the model, you can run `model.py` directly (ensure data paths are correct in the `__main__` block):
```bash
python model.py
```
