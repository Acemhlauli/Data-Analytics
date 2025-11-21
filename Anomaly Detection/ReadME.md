# Anomaly Detection Model

This repository contains a Jupyter Notebook (`EDAB_Anomaly_Detection.ipynb`) that implements and evaluates an anomaly detection model on time-series data. The goal is to accurately identify anomalous points within various datasets.

## Project Overview

The notebook develops an ensemble-based anomaly detection model that combines multiple machine learning algorithms and statistical methods. It features comprehensive data loading, feature engineering, model training, and evaluation, ultimately aiming for high accuracy in pinpointing anomalies.

## Evaluation Metric

The model's performance is evaluated based on its accuracy in identifying the single most anomalous point in a given test series. The final score is calculated as follows:

`Your Mark = (Final Evaluation % - 85) / 15 * 100`

For example, a 92% final accuracy translates to a `(92 - 85) / 15 * 100 = 46%` for the assignment.

## Data

The data consists of multiple time-series datasets, each split into a training set and a testing set. Each dataset contains a `Value1` column (the time-series data) and a `Labels` column (indicating anomalies, where `1` denotes an anomaly and `0` denotes normal data).

The data is automatically downloaded and extracted when the notebook is run.

## Model (`AnomalyDetectionModel`)

The core of this project is the `AnomalyDetectionModel` class, located in the `TAUwkbito0GL` cell. This model employs a sophisticated ensemble approach:

- **Feature Engineering**: It generates various time-series features such as rolling means, standard deviations, slopes, accelerations, and residuals to capture temporal patterns and deviations.
- **Ensemble Learning**: It combines predictions from three powerful anomaly detection algorithms:
    - **Isolation Forest**: A tree-based algorithm effective for isolating anomalies.
    - **Local Outlier Factor (LOF)**: A density-based algorithm that measures local deviation of density of a given data point with respect to its neighbours.
    - **One-Class SVM (OCSVM)**: A support vector machine that learns a decision boundary for the normal class and detects anomalies as outliers.
- **Statistical Score**: Incorporates a local Z-score to measure deviations from recent values.
- **Weighted Averaging**: The scores from individual models and the statistical score are combined using optimised weights (determined through grid search).
- **Score Smoothing**: Applies a rolling mean to the final anomaly scores to reduce noise and emphasise significant anomalies.

### Model Parameters (tuned for 94% accuracy):

```python
# Best parameters from grid search (94% accuracy)
self.iso_n_estimators = 300
self.iso_contamination = 0.01
self.lof_n_neighbors = 15
self.svm_nu = 0.03
self.svm_gamma = 'scale'
self.iso_weight = 0.4
self.lof_weight = 0.2
self.svm_weight = 0.1
self.rolling_window = 7
self.smooth_window = 2
```

## Usage

To run the notebook and evaluate the model:

1. Open the `EDAB_Anomaly_Detection.ipynb` notebook in Google Colab or any Jupyter environment.
2. Run all cells in sequence.
3. The data download, model training, and evaluation will be performed automatically.
4. The final accuracy score will be printed at the end.

**Note**: You are only allowed to edit the code within the designated `AnomalyDetectionModel` cell (`TAUwkbito0GL`).

## Visualizations

The notebook includes several helper functions for visualisation, which are useful for understanding the data and model performance:

- `visualize_dataset_overview`: Provides statistics and plots for all datasets.
- `visualize_distributions`: Shows the distribution of normal and anomalous values.
- `visualize_multiple_series`: Plots multiple time series with anomalies.
- `visualize_single_series`: Detailed view of a single time series.
- `visualize_model_scores`: Compares anomaly scores from different models.
- `visualize_features`: Illustrates engineered features.

These visualisations help in diagnosing model behaviour and understanding the nature of anomalies.

## Results

The model achieved a **94% accuracy** on the provided test datasets, exceeding the 85% baseline.

```
Total score: 94%
```

This corresponds to a score of `(94 - 85) / 15 * 100 = 60%` for the assignment.
