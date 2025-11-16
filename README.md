
# **Ayanda Mhlauli: Data Analytics Portfolio**

[![Portfolio](https://img.shields.io/badge/Portfolio-Live%20Site-blue)](https://sites.google.com/view/mhlauliayanda-eportfolio/home?authuser=0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![UFS-ZA BASE](https://img.shields.io/badge/Python-3.8%2B-blue)]([https://www.python.org/](https://github.com/ufs-za/BASE))

---

## Overview

This repository contains the complete technical implementation for a series of advanced analytics projects completed as part of the **B.Com Honours in Business and Financial Analytics (EDAB 6808)** at the University of the Free State.

The work presented integrates **economics, finance, data science, forecasting, machine learning, Bayesian inference, and natural language processing**, and demonstrates the application of scientific methodologies to real-world business problems.

---

## Live ePortfolio

A curated version of this portfolio, including narrative explanations, visual summaries, project insights, academic reflections, and professional documentation, is available at:

### **[Live ePortfolio](https://sites.google.com/view/mhlauliayanda-eportfolio/home?authuser=0)**

The ePortfolio includes:

* Executive summaries of each project
* Interactive visualisations and dashboards
* Academic reflections
* Professional profile (CV, LinkedIn, GitHub)

---

## Repository Structure

```
Ayanda-Mhlauli-Portfolio/
│
├── 01-Stock-Market-Predictions/
│   ├── notebooks/
│   ├── data/
│   ├── output/
│   └── README.md
│
├── 02-Bayesian-Networks/
│   ├── notebooks/
│   ├── data/
│   ├── models/
│   └── README.md
│
├── 03-Demographic-Profiling/
│   ├── notebooks/
│   ├── output/
│   └── README.md
│
├── 04-Discourse-Sentiment-Mapping/
│   ├── notebooks/
│   ├── output/
│   └── README.md
│
├── LICENSE
└── README.md
```

---

## Projects Overview

### 1. Stock Market Predictions Using Google Trends

**Folder:** `/01-Stock-Market-Predictions`

**Objective:** Evaluate whether Google Search Trends can serve as a leading indicator of FirstRand stock price movements on the Johannesburg Stock Exchange.

**Methods Used:**

* Time-series econometrics: ADF tests, Granger causality
* Machine learning: XGBoost, Random Forest, KNN
* Feature engineering: lag structures, moving averages
* Correlation and cross-correlation analysis

**Key Finding:**
Google Trends data for financial keywords provides statistically significant predictive information (p < 0.05), with XGBoost producing the strongest forecasting accuracy.

---

### 2. Bayesian Networks and Synthetic Data Generation

**Folder:** `/02-Bayesian-Networks`

**Objective:** Investigate how synthetic data generation methods impact Bayesian Network performance on imbalanced multi-target consumer behaviour data.

**Methods Used:**

* Bayesian Network construction (DAG modelling, probabilistic inference)
* Imbalanced data remedies: SMOTE, KNN-based synthesis, GANs
* Feature selection: Random Forest importance, Pearson correlations
* Variable Elimination for inference

**Key Finding:**
KNN-based synthetic generation achieved the strongest predictive accuracy (0.757) and stability across models. The project demonstrates the mathematical necessity of training separate models for each correlated purchase intention variable.

---

### 3. Demographic Profiling Using Bayesian Inference

**Folder:** `/03-Demographic-Profiling`

**Objective:** Identify demographic profiles most likely to exhibit specific purchase intentions using validated Bayesian Networks from Project 2.

**Methods Used:**

* Conditional probability queries
* Segmentation analysis
* Demographic scenario testing

**Key Finding:**
Twelve high-probability consumer segments were identified, providing actionable insight for targeted marketing strategies.

---

### 4. Discourse and Sentiment Mapping

**Folder:** `/04-Discourse-Sentiment-Mapping`

**Objective:** Analyse public discourse from confidential social media data to map sentiment and identify dominant thematic clusters.

**Methods Used:**

* Text preprocessing: tokenisation, lemmatisation, stopword filtering
* Sentiment analysis: VADER, TextBlob
* Topic modelling: LDA with coherence validation
* Interactive visualisation: pyLDAvis

**Key Finding:**
Five thematic topics with distinct sentiment patterns were uncovered, offering strategic insights for communication management.

**Ethical Notice:**
Raw data remains confidential under NDA and is excluded from this repository.

---

## Technical Skills Demonstrated

| Domain                  | Tools & Techniques                                            |
| ----------------------- | ------------------------------------------------------------- |
| Statistical Modelling   | ARIMA, ADF, Granger causality, hypothesis testing             |
| Machine Learning        | XGBoost, Random Forest, KNN, feature engineering              |
| Deep Learning           | GANs, neural architectures (PyTorch)                          |
| Probabilistic Modelling | Bayesian Networks, DAGs, Variable Elimination                 |
| NLP                     | Sentiment analysis (VADER, TextBlob), LDA, text preprocessing |
| Data Engineering        | External API integration, data cleaning, transformation       |
| Visualisation           | matplotlib, seaborn, plotly, pyLDAvis, networkx               |
| Programming             | Python, Jupyter Notebooks, Git/GitHub                         |

---

## Achievements

* Selected as a **GradStar Top 100** student in South Africa
* Represented UFS in the **2025 CFA Research Challenge**
* Member of the **CMDP Programme**
* Recognised as top 15 in the Department of Economics and Finance
* Academic and applied projects reviewed by supervisors and practitioners

---

## About Me

I am a data analytics specialist with a strong foundation in economics, finance, and computational methods. My analytic interests include predictive modelling, causal inference, machine learning, and applied econometrics.

**Professional Links:**

* **Email:** [Email Mhlauli Ayanda](mailto:ayandamhlauli0@gmail.com)
* **LinkedIn:** [Visit my LinkedIn](https://www.linkedin.com/in/mhlauliayanda/)
* **GitHub:** [Visit my GitHub](https://github.com/Acemhlauli)
* **ePortfolio:** [Visit my ePortfolio](https://sites.google.com/view/mhlauliayanda-eportfolio/home?authuser=0)
* **CV:** [View my CV](https://drive.google.com/file/d/1bGdTIFMwfU4tw0T7eQMpLAjBo3olInH9/view?usp=sharing)

---

## Academic Context

**Programme:** B.Com Honours in Business and Financial Analytics
**Module:** EDAB 6808
**Institution:** University of the Free State
**Supervisor:** Dr Herkulaas MvE Combrink, and Nyashadzashe Tamuka
**Year:** 2025

---

## Installation and Usage

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or Google Colab
```

### Clone the Repository

```bash
git clone https://github.com/Acemhlauli/data-analytics-portfolio.git
cd data-analytics-portfolio
```

### Install Dependencies

Each project contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Run Notebooks

```bash
jupyter notebook
```

---

## License

This portfolio is distributed under the MIT License.
See the LICENSE file for further information.

---

## Acknowledgements

* Dr. Herkulaas MvE Combrink and Nyashadzashe Tamuka for supervision and guidance
* University of the Free State for academic support
* EDAB 6808 cohort for peer insights
* Open-source developers for tools, libraries, frameworks, and OpenAI and other LLMs

---

## Contact

For academic or professional inquiries:

* **Email:** [Email Mhlauli Ayanda](mailto:ayandamhlauli0@gmail.com)
* **LinkedIn:** [Visit my LinkedIn](https://www.linkedin.com/in/mhlauliayanda/)
* **GitHub:** [Visit my GitHub](https://github.com/Acemhlauli)
* **ePortfolio:** [Visit my ePortfolio](https://sites.google.com/view/mhlauliayanda-eportfolio/home?authuser=0)
* **CV:** [View my CV](https://drive.google.com/file/d/1bGdTIFMwfU4tw0T7eQMpLAjBo3olInH9/view?usp=sharing)
* **GitHub:** [Visit UFS BASE GitHub]([https://github.com/Acemhlauli](https://github.com/ufs-za/BASE))
---

**Last Updated:** November 2025
**Repository Status:** Complete and actively maintained.

---

