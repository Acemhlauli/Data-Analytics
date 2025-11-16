# 🧠 **Assignment 3 — Demographic Profiling of Purchase Intent Using Probabilistic Models**

### **EDAB 6808 — Business & Financial Analytics (Hons)**

### University of the Free State

---

## 📌 **Overview**

This repository contains all files, code, and outputs for **Assignment 3: Demographic Profiling of Purchase Intent Using Probabilistic Models**.
The project extends the Bayesian Network modelling work from **Assignment 2**, incorporating both **imbalanced** and **synthetically balanced** datasets (SMOTE, ML-based, and GAN-based augmentation).

The primary goal is to identify which combinations of **demographic variables** are most predictive of each of the four **purchase intention** categories found in the *Subsistence Retail Consumer Dataset* (Zulu & Nkuna, 2022).

This repo forms part of the final **ePortfolio** for EDAB6808 and demonstrates advanced skills in probabilistic modelling, inference, data balancing, and consumer insights.

---

## 🎯 **Assignment Objective**

> **To determine which combinations of demographic variables are most likely to result in each of the four purchase intention types by analysing both balanced and unbalanced Bayesian networks.**
> (Based on the paper: *Data modelling of subsistence retail consumer purchase behaviour in South Africa* — Zulu & Nkuna, 2022)

---

## 📂 **Repository Structure**

```
📦 Assignment-3-Demographic-Profiling
│
├── 📁 data/
│   ├── Subsistence_Retail_Consumer_Data.csv
│   ├── Original_bayesian_models.pkl
│   ├── SMOTE_bayesian_models.pkl
│   ├── ML_bayesian_models.pkl
│   ├── GAN_bayesian_models.pkl
│
├── 📁 notebooks/
│   └── Mhlauli_A-Assignment_Three_Demographic_Profiling.ipynb
│
├── 📁 output/
│   ├── Ayanda_Assignment3_Presentation.pptx
│   ├── Rankings_Tables.csv
│   └── Inference_Results.csv
│
└── README.md
```

---

## 🗂 **1. Data Context**

### **Dataset**

* *Subsistence Retail Consumer Data*
* Collected in South Africa to understand consumer motivations in township/subsistence retail environments
* Variables include:

  * Gender
  * Age
  * Marital Status
  * Level of Education
  * Employment Status
  * Regular Customer Type
  * Shopping Frequency
  * Four Purchase Intentions (PI1–PI4)

### **Models Used**

* **Original Bayesian Models** (imbalanced data)
* **SMOTE-balanced models**
* **ML-balanced models**
* **GAN-balanced models**

---

## 🧩 **2. Methodology & Process**

### **Step 1: Load Trained Models**

All `.pkl` Bayesian Networks from Assignment 2 are loaded and combined into a unified model dictionary.

### **Step 2: Demographic Variable Extraction**

Each model contains different demographic nodes → these are automatically detected to avoid invalid inference.

### **Step 3: Generate Demographic Combinations**

For every model:

* Identify demographic variables available
* Generate **all plausible combinations** using Cartesian products
* Total combinations generated: **1,215**

### **Step 4: Bayesian Inference**

Using **Variable Elimination**, conditional probabilities are computed:

[
P(PI_i \mid \text{Demographic Combination})
]

For each intention state (1–5).

### **Step 5: Ranking & Interpretation**

Top demographic profiles are ranked by:

* Highest probability
* Intention state
* Model type
* Entropy

### **Step 6: Visualization**

* Decision tables
* Ranked probability trees
* Heatmaps
* Comparison summaries between balanced vs. unbalanced networks

---

## 📊 **3. Key Insights**

### **Across All Models**

* **Shopping frequency** is the strongest predictor of purchase intention.
* **Education level** significantly affects PI2–PI4.
* Synthetic models (SMOTE, ML, GAN) reduce bias by redistributing intention probabilities.
* **GAN models** emphasize “Regular Customer” behaviour more strongly than others.

### **Bias Observations**

* Original models overweight high-frequency shoppers → model bias.
* Balanced models smooth extreme probabilities and reveal more equitable patterns.
* Demographic bias is significantly reduced through balancing.

### **High-Probability Segments**

* Employed/diploma consumers shopping 2–3 times/week
* Young regular shoppers (GAN models)
* Occasional low-frequency shoppers with high satisfaction

### **At-Risk / Low Scores**

* High-frequency shoppers (fatigue effect)
* Highly educated consumers (higher expectations)

---

## 🧭 **4. How to Run This Project**

### **Option 1 — Google Colab (Recommended)**

1. Upload `.pkl` model files and dataset
2. Open the notebook:

   ```
   notebooks/Mhlauli_A-Assignment_Three_Demographic_Profiling.ipynb
   ```
3. Run cells sequentially — all dependencies install automatically.

### **Option 2 — Local Python Environment**

Install requirements:

```bash
pip install pandas numpy pgmpy matplotlib seaborn bnlearn networkx openpyxl
```

Run notebook via VSCode/Jupyter:

```bash
jupyter notebook
```

---

## 🎤 **5. Presentation File**

PowerPoint presentation summarising findings:

📌 `output/Ayanda_Assignment3_Presentation.pptx`

Includes:

* Research problem
* Methods
* Key visuals
* Interpretation
* Recommendations

---

## 📝 **6. Additional Notes for the ePortfolio**

* This repository should be linked under **Assignment 3** on your Google Site.
* Include the following:
  ✔ Link to this GitHub repo
  ✔ Link to Google Colab notebook
  ✔ Screenshot of probability tables/heatmaps
  ✔ Short reflection:

  * What you learned
  * Challenges with probabilistic modelling
  * Insights gained from balancing data

This helps you score high in the **Technical Integration** and **Content Organisation** categories of the rubric.

---

## 📚 **7. References**

Zulu, N., & Nkuna, N. (2022). *Data modelling of subsistence retail consumer purchase behaviour in South Africa.*
Sun, B., & Morwitz, V. (2010). *Stated intentions and purchase behaviour: A unified model.*
Nawi, N. M., et al. (2019). *Effect of consumer demographics and risk factors on online purchase behaviour.*
Luo, X. (2016). *Demographic-related purchase behaviours.*
pgmpy Documentation
bnlearn Documentation

---

## 🧑‍💻 **8. Author**

**Ayanda Mhlauli**
B.Com Honours – Business & Financial Analytics
University of the Free State

---

If you'd like, I can also generate:

✅ A shorter, public-friendly README
✅ A GitHub profile README
✅ A markdown version for your Google Site
Just tell me!
