# Model Architecture & Data Analysis Report

## 1. Dataset Analysis

Based on the preprocessing of `NEUSTG_19502020_12stations.mat`, we have established the following data scale:

*   **Training Samples:** 226,134 sequences
*   **Validation Samples:** 74,277 sequences
*   **Input Feature Space:** 28 Dimensions
    *   Derived from 7-day historical window
    *   4 features per day (Mean, Max, Min, Std of Standardized Distance)
*   **Target Output Space:** 14 Dimensions
    *   Binary classification (Flood/No-Flood) for the next 14 days

### Token/Data Point Calculation
Total Input Data Points = Samples $\times$ Features
$$ 226,134 \times 28 \approx 6.33 \text{ Million Data Points} $$

## 2. Model Selection Strategy

### Assessment
The dataset is classified as **Medium-Sized Tabular**.
*   It is structured (dense numerical features), not unstructured (text/image).
*   The relationship between "Distance to Threshold" and "Flood Probability" is likely monotonic but non-linear.

### Candidate Architectures
| Architecture | Suitability | Reasoning |
| :--- | :--- | :--- |
| **Transformer (LLM/BERT)** | Low | Overkill. 6M data points is insufficient to train large attention heads from scratch without massive overfitting. The input sequence length (7) is too short to benefit from attention mechanisms. |
| **Deep ResNet / CNN** | Low | Numerical features lack the spatial correlation (pixels) that CNNs exploit. |
| **LSTM / RNN** | Medium | Good for time-series, but can be slow to train and harder to tune for simple tabular dependencies. |
| **XGBoost (GBT)** | **High** | **Chosen.** Gradient Boosted Trees are the State-of-the-Art (SOTA) for tabular data up to ~10M rows. They handle non-linear decision boundaries efficiently and require less hyperparameter tuning than Neural Networks. |

## 3. Selected Approach: Global XGBoost

We will implement a **Multi-Output XGBoost Regressor**.

*   **Input:** 28 standardized features (normalized by station variance and threshold).
*   **Output:** 14 binary probabilities (one for each forecast day).
*   **Training Strategy:**
    *   Train a single "Global" model on all 9 training stations combined.
    *   This forces the model to learn the universal physics of "Distance to Threshold" rather than memorizing specific station behavior.
    *   This directly addresses the **Out-of-Distribution (OOD)** challenge.

### Expected Size
*   **Trees:** ~100-500 trees
*   **Depth:** 4-6 (to prevent overfitting)
*   **Inference Speed:** < 10ms per batch (extremely fast)
