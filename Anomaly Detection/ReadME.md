# Anomaly Detection using Ensemble Machine Learning

An optimized ensemble-based anomaly detection system that combines multiple machine learning algorithms to identify anomalous points in time-series data with 94% accuracy.

## 📊 Overview

This project implements a multi-algorithm ensemble approach for detecting single anomalous points in time-series datasets. The system was designed to work on unseen data without prior knowledge of data characteristics, achieving 94% accuracy on diverse time-series patterns.

## 🎯 Performance

- **Training Accuracy:** 94%
- **Assignment Score:** 60% [(94-85)/15 × 100]
- **Baseline Requirement:** 85%

## 🔧 Model Architecture

The solution combines four detection components in a weighted ensemble:

| Algorithm | Weight | Purpose |
|-----------|--------|---------|
| Isolation Forest | 40% | Detects global outliers and unusual feature combinations |
| Local Outlier Factor (LOF) | 20% | Identifies contextual anomalies in local neighborhoods |
| One-Class SVM | 10% | Captures complex non-linear patterns |
| Statistical Z-Score | 10% | Flags sudden deviations from local trends |

**Final Score Formula:**
```
Anomaly Score = 0.4×ISO + 0.2×LOF + 0.1×SVM + 0.1×Z-score
```

## ✨ Features

### Feature Engineering
The model creates 8 temporal features from raw time-series data:

1. Original value
2. Rolling means (windows: 3, 7, 15)
3. Rolling standard deviations (windows: 3, 7)
4. First derivative (slope)
5. Second derivative (acceleration)
6. Residuals from 7-point moving average

### Hyperparameter Optimization
Grid search across 39,366 parameter combinations optimized:
- Contamination rates: 0.01-0.03
- Estimators: 300-700
- Neighbors: 15-25
- SVM nu values: 0.03-0.07
- Window sizes: 2-15 points

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/anomaly-detection-ensemble.git
cd anomaly-detection-ensemble

# Install required packages
pip install -r requirements.txt
```

### Requirements
```
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
```

## 💻 Usage

### Basic Usage
```python
from anomaly_detection import AnomalyDetectionModel
import pandas as pd

# Load your data
train_data = pd.read_csv('train.csv', sep=';')
test_data = pd.read_csv('test.csv', sep=';')

# Initialize and train model
model = AnomalyDetectionModel()
model.fit(train_data.Value1.values, train_data.Labels.values)

# Predict anomaly index
anomaly_index = model.predict(test_data.Value1.values)
print(f"Anomaly detected at index: {anomaly_index}")
```

### Evaluation
```python
# Evaluate on multiple datasets
correct = 0
for train, test in zip(train_files, test_files):
    model = AnomalyDetectionModel()
    model.fit(train.Value1.values, train.Labels.values)
    prediction_index = model.predict(test.Value1.values)
    
    if test.loc[prediction_index, "Labels"] == 1:
        correct += 1

print(f"Total score: {correct}%")
```

## 📈 Visualizations

The project includes comprehensive visualization tools:
```python
from visualizations import (
    visualize_single_series,
    visualize_model_scores,
    visualize_distributions,
    visualize_features
)

# Visualize a single dataset with anomalies
visualize_single_series(train_df, test_df, dataset_idx=0)

# Compare model scores
visualize_model_scores(train_df, test_df, dataset_idx=0)

# Analyze distributions
visualize_distributions(train_files, test_files)

# View engineered features
visualize_features(test_df, dataset_idx=0)
```

## 🔬 Methodology

### Ensemble Strategy
The system leverages complementary strengths:
- **Isolation Forest:** Global outlier detection
- **LOF:** Contextual anomaly identification
- **SVM:** Pattern-based boundary detection
- **Z-scores:** Statistical deviation measurement

### Key Success Factors
1. **Ensemble Diversity:** Multiple algorithm types prevent failure on specific patterns
2. **Multi-scale Features:** Windows of 3, 7, and 15 points capture different temporal contexts
3. **Optimized Weighting:** Grid search-derived weights balance each method's contribution
4. **Score Smoothing:** 2-point rolling window reduces false positives
5. **Low Contamination:** 0.01 rate reflects real-world anomaly rarity

## 📊 Project Structure
```
anomaly-detection-ensemble/
│
├── anomaly_detection.py       # Main model implementation
├── visualizations.py           # Visualization functions
├── grid_search.py             # Hyperparameter optimization
├── requirements.txt           # Package dependencies
├── README.md                  # This file
│
├── notebooks/
│   ├── EDAB6808_Anomaly_Detection.ipynb
│   └── analysis_and_visualization.ipynb
│
└── data/
    ├── train/                 # Training datasets
    └── test/                  # Test datasets
```

## 🎯 Applications

This anomaly detection system can be applied to:

- **Financial Markets:** Detecting unusual trading patterns or price movements
- **Manufacturing:** Identifying sensor anomalies indicating equipment failure
- **Cybersecurity:** Flagging abnormal network behavior
- **Healthcare:** Detecting unusual patient vital signs requiring intervention
- **IoT Systems:** Monitoring sensor data for unexpected readings

## 🔍 Key Findings

### Model Performance
- Isolation Forest excels at global outliers (extreme values)
- LOF performs best on contextual anomalies (local pattern deviations)
- SVM captures complex non-linear boundaries
- Z-scores provide interpretable statistical validation

### Feature Importance
- **Slope (1st derivative):** Highly discriminative for rapid changes
- **Acceleration (2nd derivative):** Captures sudden direction changes
- **Rolling means:** Smooth noise while preserving signals
- **Standard deviations:** Highlight volatility-based anomalies

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@software{anomaly_detection_ensemble,
  author = {Your Name},
  title = {Anomaly Detection using Ensemble Machine Learning},
  year = {2024},
  url = {https://github.com/yourusername/anomaly-detection-ensemble}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact
Project Link: [https://github.com/yourusername/anomaly-detection-ensemble](https://github.com/yourusername/anomaly-detection-ensemble)

## 🙏 Acknowledgments

- EDAB6808 course for the project framework
- Scikit-learn library for machine learning implementations
- The anomaly detection research community

---

⭐ If you found this project helpful, please consider giving it a star!
