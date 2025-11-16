# Discourse & Sentiment Mapping

## Project Overview

This repository contains the full implementation for an analysis of a large, confidential social media dataset, as per the EDAB 6808 Assignment 4 brief. The project's objective was to move beyond simple engagement metrics to map the underlying structure of the online discourse.

This was achieved by integrating sentiment analysis and topic modelling to answer the central research question: **What is the overall emotional tone and thematic structure of this online discourse, and is the conversation constructive or polarised?**

## ⚠️ Data & Ethics Notice (NDA)

This project was conducted under a **strict Non-Disclosure Agreement (NDA)**.

* The raw dataset is **confidential** and **cannot** be shared or included in this repository.
* All analysis was performed at an **aggregate level** to protect participant privacy and anonymity.
* The findings presented in the notebook and this README are anonymised and focus on high-level strategic insights, in full compliance with the NDA.

## Table of Contents

* [Repository Structure](#repository-structure)
* [Methodology](#methodology)
* [Key Results](#key-results)
* [Installation & Requirements](#installation--requirements)
* [Usage Instructions](#usage-instructions)
* [Contributors](#contributors)
* [License](#license)

## Repository Structure

```
Discourse-and-Sentiment-Mapping/
│
├── notebooks/
│   └── Group_3_Mhlauli_A_&_Matlhare_T.ipynb
│
├── data/
│   └── .gitkeep (This folder is intentionally empty per the NDA)
│
├── README.md
└── LICENSE
```

## Methodology

The analysis was conducted using a 4-phase NLP pipeline, as detailed in the project notebook:

1. **Data Preprocessing:** The raw text was extensively cleaned. This included lowercasing, removing URLs, mentions, and hashtags, eliminating stopwords (using `nltk`), and performing lemmatisation (using `spacy`) to normalise the text.

2. **Sentiment Analysis:** The **VADER** (Valence Aware Dictionary and sEntiment Reasoner) model was used to assign a compound, positive, negative, and neutral sentiment score to each cleaned message.

3. **Topic Modelling:** **Latent Dirichlet Allocation (LDA)** (using `gensim`) was applied to the preprocessed corpus to identify and extract the dominant, hidden themes from the discourse.

4. **Integrated Analysis:** The sentiment scores and topic distributions were merged, allowing for an analysis of the average sentiment *per* topic.

## Key Results

The analysis provided a clear answer to the research question: **the discourse is stable, constructive, and overwhelmingly positive**.

* **1. Overwhelmingly Positive Sentiment:** The discourse was **63.19% positive**, compared to only **11.08% negative**. This indicates a highly favourable and supportive community environment, not a platform for complaints.

* **2. Seven Dominant Themes Identified:** The LDA topic model successfully identified **seven** main themes, including "Business, Media & Finance" and "Education & Learning." The largest and most central theme was **"General Discussion,"** which comprised nearly 24% of all posts.

* **3. Complete Absence of Polarisation:** A significant finding was the **complete absence of polarisation**. When mapping sentiment to topics, every single one of the seven themes registered a "→ MIXED" sentiment profile. This confirms the discourse is stable and not fragmented into polarised blocks.

## Installation & Requirements

The project notebook was run in a Google Colab environment. The following libraries are required:

```bash
# Install required libraries
pip install pandas numpy nltk spacy
pip install vaderSentiment
pip install gensim
pip install pyLDAvis
pip install --upgrade scikit-learn
pip install matplotlib seaborn
```

**Additional Setup:**

```python
# Download required NLTK data
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Download spaCy language model
!python -m spacy download en_core_web_sm
```

## Usage Instructions

1. **Data is Not Provided:** Due to the NDA, the source data file is not and cannot be included in this repository.

2. **Review Analysis:** The full analytical process, from preprocessing to visualisation, can be reviewed end-to-end in the `notebooks/Group_3_Mhlauli_A_&_Matlhare_T.ipynb` file. All outputs, charts, and interpretations are saved within the notebook.

## Contributors

* Ayanda Mhlauli
* Tshenolo Matlhare

### Course

* **Course:** B.Com Honours in Business and Financial Analytics (EDAB 6808)
* **Institution:** University of the Free State
* **Supervisor:** Dr. Herkulaas MvE Combrink

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
