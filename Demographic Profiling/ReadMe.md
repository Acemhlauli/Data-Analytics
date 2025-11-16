# **Demographic Profiling of Purchase Intent Using Probabilistic Models**

## **Project Overview**

This repository contains the full implementation and analysis for **probabilistic modelling of purchase intentions** using Bayesian Networks trained on both imbalanced and synthetically balanced datasets. The project aims to identify **which demographic combinations** most strongly predict each of the four purchase intention outcomes (PI1–PI4) within subsistence retail contexts in South Africa.

**Research Question:**
How do demographic variables—combined in all plausible configurations—affect the conditional probability of each purchase intention outcome, and how do different balanced datasets (SMOTE, ML-based, and GAN-based) influence the stability, fairness, and interpretability of Bayesian inference?

**Key Innovation:**
An exhaustive evaluation of **1,215 valid demographic combinations** across **16 Bayesian models**, using Variable Elimination to compute conditional probabilities:
[
P(PI_i \mid \text{Demographic Combination})
]
This approach enables deep insight into demographic sensitivity, model bias, balancing effects, and consumer behavioural heterogeneity.

---

## **Table of Contents**

1. [Dataset](#dataset)
2. [Repository Structure](#repository-structure)
3. [Methodology](#methodology)
4. [Key Results](#key-results)
5. [Installation & Requirements](#installation--requirements)
6. [Usage Instructions](#usage-instructions)
7. [Reproducibility](#reproducibility)
8. [Contributors](#contributors)
9. [References](#references)
10. [License](#license)

---

## **Dataset**

**Source:** Subsistence Retail Consumer Purchase Behaviour Dataset (Zulu & Nkuna, 2022)
**Sample Size:** 281 consumers
**Variables:**

* **7 Demographic features** (Gender, Age, Marital Status, Employment, Education, Regular Customer, Shopping Frequency)
* **4 Purchase Intention variables** (PI1–PI4), all measured on 5-point Likert scales
* **Measurement Scales:** Service quality, trust, convenience, product perception, etc.

**Purchase Intention Class Distributions:**

```
PI1: {1: 4,  2: 24,  3: 57,  4: 136, 5: 60}
PI2: {1: 11, 2: 15,  3: 58,  4: 123, 5: 74}
PI3: {1: 11, 2: 27,  3: 53,  4: 119, 5: 71}
PI4: {1: 7,  2: 23,  3: 55,  4: 123, 5: 73}
```

**Statistical Challenges:**

* Severe class imbalance (ratios up to 34:1)
* Mixed categorical/ordinal data
* Four correlated outcomes (PI1–PI4)
* Risk of demographic bias affecting inference

---

## **Repository Structure**

```
Assignment-3-Demographic-Profiling/
│
├── data/
│   ├── Subsistence_Retail_Consumer_Data.csv
│   ├── Original_bayesian_models.pkl
│   ├── SMOTE_bayesian_models.pkl
│   ├── ML_bayesian_models.pkl
│   └── GAN_bayesian_models.pkl
│
├── notebooks/
│   └── Mhlauli_A_Assignment_Three_Demographic_Profiling.ipynb
│
├── output/
│   ├── Inference_Results.csv
│   ├── Rankings_Tables.csv
│   └── Ayanda_Assignment3_Presentation.pptx
│
├── src/ (optional modularisation)
│   ├── inference_engine.py
│   ├── combination_generator.py
│   ├── model_loader.py
│   └── balancing_utils.py
│
├── README.md
└── LICENSE
```

---

## **Methodology**

The analysis consisted of four structured phases.

### **Phase 1: Model Preparation and Combination Generation**

All pre-trained Bayesian Networks from Assignment 2—Original, SMOTE-balanced, ML-balanced, and GAN-balanced—were loaded and indexed by purchase intention. Demographic nodes present in each model were automatically detected, after which all **valid demographic combinations** were generated using Cartesian products. A total of **1,215 plausible combinations** were created across all model groups and validated against DAG structures to ensure compatibility.

### **Phase 2: Bayesian Inference**

For each demographic combination and each PI model, conditional probability distributions were computed using **Variable Elimination**:
[
P(PI_i \mid \text{Gender}, \text{Age}, \text{Education}, ... ).
]
The algorithm yielded full probability vectors across Likert states 1–5, accompanied by entropy scores, certainty analysis, and comparison tables across balancing methods (Original, SMOTE, ML, GAN).

### **Phase 3: Ranking and Demographic Pattern Identification**

The results were ranked by maximum-probability intention state, distribution extremity, and cross-model consistency. Differences introduced by synthetic balancing were quantified to identify which demographic drivers remained stable versus those sensitive to dataset augmentation.

### **Phase 4: Synthesis and Interpretation**

Inference outputs were consolidated into CSV files and interpreted across model types, PI outcomes, and demographic categories. The analysis emphasised behavioural insights, model bias, and fairness improvements.

---

## **Key Results**

### **1. Shopping Frequency is the Most Predictive Variable**

Across all models, shopping frequency overwhelmingly dominated as the strongest predictor of PI outcomes. Consumers shopping **2–3 times per week** consistently showed higher positive intentions, while **6–7 times per week** corresponded to lower scores (fatigue effect).

### **2. Education Significantly Shapes PI2–PI4**

Higher education levels frequently produced lower intention scores, whereas diploma/basic education groups consistently appeared in top-ranked profiles for positive intentions.

### **3. Balanced Models Improve Fairness and Stability**

SMOTE, ML, and GAN balancing reduced structural bias by smoothing probability distributions and improving the representation of youth and regular-customer segments. GAN models demonstrated particularly strong behavioural sensitivity.

### **4. High-Probability Segments Identified**

* Diploma/basic-education consumers
* Shoppers with 2–3 visits weekly
* Young regular customers (GAN models)
* Employed moderate-frequency shoppers

### **5. Methodological Limitations**

* Rare demographic edges produced unstable inference
* DAG constraints inherited from Assignment 2
* Balancing approaches affect demographic representation differently

---

## **Installation & Requirements**

### **Conda Environment (Recommended)**

```bash
conda create -n assignment3 python=3.9
conda activate assignment3
pip install -r requirements.txt
```

### **Required Libraries**

```
pandas
numpy
pgmpy
bnlearn
scikit-learn
matplotlib
seaborn
networkx
```

---

## **Usage Instructions**

### **Run the Full Inference Pipeline**

```bash
jupyter notebook notebooks/Mhlauli_A_Assignment_Three_Demographic_Profiling.ipynb
```

### **Example: Computing Inference**

```python
from src.inference_engine import run_inference

prob_dist = run_inference(
    model=original_models['PI1'],
    combination={
        'Gender': 'Male',
        'Education': 'Diploma',
        'Shopping_Frequency': '2-3_times/week'
    }
)
print(prob_dist)
```

### **Exporting Ranked Results**

```python
inference_df.to_csv('output/Inference_Results.csv', index=False)
rankings_df.to_csv('output/Rankings_Tables.csv', index=False)
```

---

## **Reproducibility**

### **Random Seeds**

```python
import numpy as np
np.random.seed(42)
```

### **Model Loading**

```python
import pickle
with open('data/Original_bayesian_models.pkl', 'rb') as f:
    original_models = pickle.load(f)
```

### **Notebook-Based Reproducibility**

All analytical steps are executable end-to-end inside the notebook.

---

## **Contributors**

**Student:**

* **Ayanda Mhlauli** — Inference engine implementation, demographic profiling analysis, probability ranking, presentation development.

**Course:**

* EDAB 6808: Business and Financial Analytics (Honours)
* University of the Free State

**Supervisor:**

* Dr. Herkulaas MvE Combrink

---

## **References**

Zulu, N., & Nkuna, N. (2022). Data modelling of subsistence retail consumer purchase behaviour in South Africa. *Data in Brief, 42*, 108066.
Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.*
Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models.*

---

## **License**

This project is licensed under the MIT License.
Academic reuse requires proper citation.

---

