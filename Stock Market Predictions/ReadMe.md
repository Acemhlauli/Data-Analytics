# **Lead-Lag Indicators in Stock Market Predictions**

## Project Overview

This repository contains the full implementation and analysis for an investigation into the predictive power of Google Search trends on stock market movements. The project uses time-series analysis and machine learning to determine if Google Search Volume Index (GSVI) data can serve as a leading or lagging indicator for the stock price of **FirstRand (FSR)**, a company listed on the Johannesburg Stock Exchange (JSE).

### Research Question:

  * Can Google Search Volume Index (GSVI) data serve as a statistically significant leading or lagging indicator for the daily stock price movements of FirstRand (FSR) on the JSE?

### Key Methodological Approach:

This project integrates classical time-series econometrics with modern machine learning regression. A vector of lagged GSVI features and moving averages was engineered and tested using:

1.  **Statistical Tests:** Cross-Correlation and Granger Causality to identify the existence and timing of predictive relationships.
2.  **Machine Learning:** Predictive models (KNN, Random Forest, XGBoost) to quantify the predictive power and directional accuracy of these relationships.

The core predictive model attempts to solve:
`Price_diff(t) = α + Σ[β_i * Price_diff(t-i)] + Σ[γ_j,k * Keyword_j(t-k)] + ε`

## Table of Contents

  * [Dataset](https://www.google.com/search?q=%23dataset)
  * [Repository Structure](https://www.google.com/search?q=%23repository-structure)
  * [Methodology](https://www.google.com/search?q=%23methodology)
  * [Key Results](https://www.google.com/search?q=%23key-results)
  * [Installation & Requirements](https://www.google.com/search?q=%23installation--requirements)
  * [Usage Instructions](https://www.google.com/search?q=%23usage-instructions)
  * [Reproducibility](https://www.google.com/search?q=%23reproducibility)
  * [Contributors](https://www.google.com/search?q=%23contributors)
  * [References](https://www.google.com/search?q=%23references)
  * [License](https://www.google.com/search?q=%23license)

## Dataset

Data was collected from two primary, dynamic sources for a one-year period (May 2023 - April 2024).

1.  **Stock Market Data:**

      * **Source:** `yfinance` Python library.
      * **Ticker:** `FSR.JO` (FirstRand Ltd.) and `^J203.JO` (JSE All Share Index).
      * **Variable:** Daily Closing Price.

2.  **Search Interest Data:**

      * **Source:** `pytrends` (Google Trends) Python library.
      * **Keywords:** 14 search terms related to FirstRand, its subsidiaries, and broader economic indicators (e.g., "FirstRand", "RMB", "WesBank", "JSE All Share", "Interest rates South Africa").
      * **Variable:** Daily Search Volume Index.

### Statistical Challenges:

  * **Non-Stationarity:** Financial time-series data is non-stationary by nature, requiring differencing (as confirmed by ADF tests) for use in statistical models like Granger Causality.
  * **Noise:** Google Trends data is inherently noisy and subject to sporadic, unrelated spikes.
  * **Spurious Correlation:** A high risk of identifying spurious correlations, necessitating formal causality tests (Granger) beyond simple cross-correlation.

## Repository Structure

```
Assignment-1-Stock-Prediction/
│
├── notebooks/
│   └── Group_4_Mhlauli_A_&_Mukuvari_A-Assignment_One_Stock_Prediction.ipynb
│
├── presentations/
│   └── Group_4_Mhlauli_A_&_Mukuvari_A_Final_Presentaion.pdf
│
├── data/
│   └── (Optional: Can be used to store CSV exports of the merged data)
│
├── README.md
└── LICENSE
```

## Methodology

The analysis was conducted in four distinct phases, as documented in the project notebook.

### Phase 1: Data Collection and Preprocessing

Daily stock prices for FSR and the JSE All Share Index were retrieved using `yfinance`. Daily search data for 14 keywords was retrieved using `pytrends`. The two datasets were cleaned, resampled to a consistent daily frequency, and merged into a single `pandas` DataFrame.

### Phase 2: Feature Engineering

To test for lead-lag effects, a comprehensive set of features was engineered from the GSVI data. This included:

  * **Lagged Features:** 1, 2, 3, 4, 5, 6, and 7-day lags for all 14 keywords.
  * **Moving Averages:** 7-day moving averages for all 14 keywords to smooth out short-term noise.

### Phase 3: Statistical Analysis

  * **Stationarity:** The Augmented Dickey-Fuller (ADF) test was performed on all time-series variables to check for stationarity. Non-stationary variables (like stock price) were differenced.
  * **Correlation:** A cross-correlation heatmap was generated to provide an initial visual guide to potential lead-lag relationships between lagged keywords and the stock price.
  * **Causality:** The **Granger Causality Test** was systematically run on the stationary (differenced) data to determine if the time-series of any keyword "Granger-causes" (i.e., has statistically significant predictive value for) the stock price.

### Phase 4: Predictive Modelling

  * **Model Training:** Three machine learning regression models were trained to predict the next day's closing price: **K-Nearest Neighbors (KNN)**, **Random Forest**, and **XGBoost Regressor**.
  * **Evaluation:** Models were evaluated using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared (R²)**.
  * **Feature Importance:** The feature importance plot from the best-performing model (XGBoost) was analyzed to identify the most predictive features.

## Key Results

1.  **Hypothesis Supported:** The analysis concluded that Google Trends data **contains statistically significant predictive signals** for FirstRand's stock price.
2.  **Optimal Lag Identified:** The strongest predictive relationships were found at **7-day lags and 7-day moving averages**, suggesting that changes in public interest take approximately one week to be fully reflected in the stock price.
3.  **Granger Causality Confirmed:** The Granger Causality test found significant p-values for several keywords, notably "FirstRand" itself (at a 5-day lag), confirming a true predictive relationship, not just a spurious correlation.
4.  **Model Performance:** The **XGBoost Regressor** was the best-performing model, achieving the highest R-squared and lowest error.
5.  **Feature Importance:** The XGBoost model's feature importance plot confirmed that Google Trends data was highly relevant. Keywords like **"JSE All Share\_ma7"** and **"Invest\_ma7"** were ranked as top predictors, alongside FirstRand's own lagged price data.
6.  **Directional Accuracy:** The final optimised model achieved a **71% directional accuracy**, indicating it was correct in predicting the direction (up or down) of the stock's movement 7 out of 10 times.

## Installation & Requirements

The project notebook was run in a Google Colab environment. The following libraries are required:

```bash
# Install required libraries
pip install pandas numpy yfinance pytrends matplotlib seaborn statsmodels xgboost scikit-learn
```

### Required Libraries

  * `pandas`
  * `numpy`
  * `yfinance`
  * `pytrends`
  * `matplotlib`
  * `seaborn`
  * `statsmodels` (for ADF and Granger Causality tests)
  * `sklearn` (scikit-learn)
  * `xgboost`

## Usage Instructions

To replicate the analysis, open and run the main notebook in a compatible environment (like Google Colab or a local Jupyter server).

1.  **Clone the repository (optional):**
    ```bash
    git clone https://[your-repo-url]/Assignment-1-Stock-Prediction.git
    cd Assignment-1-Stock-Prediction
    ```
2.  **Run the notebook:**
      * Open `notebooks/Group_4_Mhlauli_A_&_Mukuvari_A-Assignment_One_Stock_Prediction.ipynb`.
      * Execute the cells sequentially from top to bottom. The `yfinance` and `pytrends` libraries will fetch the data live.

## Reproducibility

  * **Random Seeds:** A random seed (`42`) is set in the notebook to ensure the train/test splits and machine learning models are reproducible.
  * **Dynamic Data:** Note that `yfinance` and `pytrends` pull live data. Running the notebook at a different date will fetch new data, which may lead to slightly different (but directionally similar) results, particularly in the most recent periods.

## Contributors

  * **Ayanda Mhlauli** — Data collection, feature engineering, statistical analysis, ML modelling, and interpretation.
  * **Anesu Mukuvari** — Data collection, feature engineering, statistical analysis, ML modelling, and interpretation.

### Course

  * **Course:** EDAB 6808: Business and Financial Analytics (Honours)
  * **Institution:** University of the Free State
  * **Supervisor:** Dr. Herkulaas MvE Combrink

## References

  * Ayala, M.J., Gonzálvez-Gallego, N. and Arteaga-Sánchez, R. (2024) ‘Google search volume index and investor attention in stock market: a systematic review’, *Financial Innovation*, 10(70).
  * Da, Z., Engelberg, J., & Gao, P. (2011). In Search of Attention. *The Journal of Finance*, 66(5), 1461–1499.
  * Wang, X., et al. (2021) ‘Learning Non-Stationary Time-Series with Dynamic Pattern Extractions’, *arXiv preprint*.
  * Xi, J. (2025) Machine Learning Using Nonstationary Data. *SSRN Working Paper*.

## License

This project is licensed under the MIT License.
