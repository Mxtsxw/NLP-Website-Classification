# Website Type Classification — NLP Project

> **Classifying web pages into 7 structural categories using TF-IDF, Latent Semantic Analysis, and HTML structural signals — achieving 97.1% accuracy with LinearSVC on held-out data.**

[Try the live Demo](https://github.com/Mxtsxw/NLP-Website-Classification/)

<img width="9334" height="5924" alt="image" src="https://github.com/user-attachments/assets/bf7f7f16-272c-46a6-8f8b-bcd7a6935006" />

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Executive Summary](#executive-summary)
3. [Project Structure](#project-structure)
4. [Part 1 — Exploratory Data Analysis](#part-1--exploratory-data-analysis)
5. [Part 2 — Feature Engineering](#part-2--feature-engineering)
6. [Part 3 — Modeling & Evaluation](#part-3--modeling--evaluation)
7. [Final Results](#final-results)
8. [Tech Stack](#tech-stack)

---

## Problem Statement

Automatically understanding the purpose of a web page, whether it is a homepage, a blog post, a search results page, or an e-commerce listing, is a foundational task in web mining, content indexing, and site auditing. This project builds a full classification pipeline on a labeled corpus of HTML pages spanning **7 structural categories**, tackling the full lifecycle from raw HTML parsing to model evaluation.

**Core challenges addressed:**
- Extracting meaningful signal from raw, noisy HTML content
- Building feature representations that capture both textual semantics and structural layout
- Comparing a dense similarity-based approach (KNN + LSA) against a sparse linear classifier (SVM + TF-IDF)
- Identifying which page types are inherently harder to distinguish

---

## Executive Summary

This project demonstrates an end-to-end NLP classification workflow structured across three notebooks. Starting from raw HTML, the analysis built and compared three feature families: sparse TF-IDF with bigrams, dense 150-dimensional LSA representations via Truncated SVD, and 8 normalized structural signals derived from HTML element counts.

A KNN classifier using cosine distance on LSA + structural features and a LinearSVC on raw TF-IDF were trained and evaluated on a held-out test set. **LinearSVC achieved 97.1% accuracy and ~0.97 macro F1**, dramatically outperforming KNN (83.3%), confirming that page type is primarily a vocabulary-driven signal rather than a structural one. SVM coefficients provide direct interpretability into which terms drive each class decision.

---

## Project Structure

```
website-classification/
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb  # Feature Representations
│   └── 03_Modeling.ipynb             # Model Training & Evaluation
├── data/
│   ├── train_texts.parquet           # Processed text corpus
│   ├── test_texts.parquet
│   ├── train_features.csv            # Structural HTML features
│   └── test_features.csv
└── models/
    ├── tfidf.pkl                     # Fitted TF-IDF vectorizer
    ├── svd150.pkl                    # Fitted SVD transformer
    ├── label_encoder.pkl
    ├── knn.pkl
    └── svm.pkl
```

---

## Part 1 — Exploratory Data Analysis

**Notebook**: `01_EDA.ipynb`

### Dataset Overview

| Property | Value |
|---|---|
| Train set | 840 pages |
| Test set | 210 pages |
| Classes | 7 |
| Missing Values | None |
| Class Balance | Approximately uniform |

**Classes**: `FAQ`, `accueil`, `blog`, `commerce`, `home`, `liste`, `recherche`

---

### Class Distribution

Prior work was done to perfectly balance the dataset across the 7 classes, meaning accuracy is a reliable metric and no class-weighting strategy is required. Each class contributes roughly 120 training examples.

---

### Key Finding 1 — Text Length Variation by Class

Page types differ substantially in raw text length. `Commerce` and `liste` pages tend to be longer due to product listings and item enumerations, while `FAQ` and `recherche` pages are more compact. This motivates including `text_len` as a structural feature.

<img width="2014" height="460" alt="image" src="https://github.com/user-attachments/assets/e6bfb1d6-a2a6-44c9-abdf-37d5e7ae1a16" />


---

### Key Finding 2 — Structural Signal Differences

HTML structural counts (`n_links`, `n_forms`, `n_inputs`, `n_tables`, `n_imgs`, `n_headings`, `n_list_items`) vary meaningfully across classes. `Commerce` pages have high `n_imgs` and `n_inputs`; `recherche` pages have high `n_forms`; `liste` pages have high `n_list_items`. These signals are non-redundant with text content.

<img width="2036" height="630" alt="image" src="https://github.com/user-attachments/assets/b95a75d9-0018-4683-9d0c-10869804ad96" />


---

### Key Finding 3 — Vocabulary Overlap and Separability

Some class pairs share overlapping vocabulary. The `blog` and `home` classes are the most similar in terms of lexical content, while `recherche` and `commerce` have more distinctive vocabularies. This foreshadows the confusion patterns observed in modeling.

<img width="2039" height="552" alt="image" src="https://github.com/user-attachments/assets/0cd7659c-fb4e-410f-9623-5efbd758d21c" />


---

### Key Finding 4 — Top Terms per Class

Per-class term frequency analysis shows strong discriminative vocabulary: `panier`, `prix`, `acheter` for commerce; `question`, `réponse` for FAQ; `résultats`, `recherche` for recherche. These domain-specific terms make the classification task well-suited to TF-IDF.

<img width="2038" height="687" alt="image" src="https://github.com/user-attachments/assets/f014b614-f757-4822-8633-52edc106994d" />

---

## Part 2 — Feature Engineering

**Notebook**: `02_Feature_Engineering.ipynb`

Three feature families were constructed and saved for the modeling notebook.

### Feature Families

| Feature Set | Dimensions | Description |
|---|---|---|
| TF-IDF (sparse) | 10,000 | Unigrams + bigrams, sublinear TF, stop words removed |
| LSA / SVD (dense) | 150 | TF-IDF reduced via Truncated SVD, L2-normalized |
| Structural (dense) | 8 | Normalized HTML element counts |
| KNN combined | 158 | LSA + Structural concatenated |

---

### TF-IDF Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 10,000 | Vocabulary cap to control dimensionality |
| `sublinear_tf` | True | log(1+tf) dampens high-frequency terms |
| `ngram_range` | (1, 2) | Bigrams capture phrases like "search results", "add to cart" |
| `stop_words` | english | Removes uninformative tokens |
| `min_df` | 2 | Excludes hapax legomena |

Resulting sparsity: >99% — confirming the need for a dimensionality-reduced alternative for distance-based methods.

---

### LSA via Truncated SVD

SVD decomposes the TF-IDF matrix into latent semantic dimensions, reducing from 10,000 to 150 components. This captures synonymy and co-occurrence patterns and produces the dense vectors required by KNN's cosine distance.

150 components explain a substantial portion of total variance, as shown in the scree plot below.

<img width="2027" height="578" alt="image" src="https://github.com/user-attachments/assets/07f47d1b-a529-4e6b-89ae-338a2de87ba2" />

The 2D SVD projection shows that most classes form distinct clusters, with some overlap between `blog` and `home`.

<img width="2039" height="552" alt="image" src="https://github.com/user-attachments/assets/0cd7659c-fb4e-410f-9623-5efbd758d21c" />

---

### Structural Features

8 HTML structural signals were extracted and normalized by dividing by 100 to bring them to a comparable scale with SVD components:

`n_links`, `n_forms`, `n_inputs`, `n_tables`, `n_imgs`, `n_headings`, `n_list_items`, `text_len`

<img width="1598" height="721" alt="image" src="https://github.com/user-attachments/assets/db415427-59b6-4446-be7a-d7d17fdf0d7e" />

---

### Saved Artifacts

| File | Description |
|---|---|
| `tfidf.pkl` | Fitted TF-IDF vectorizer |
| `svd150.pkl` | Fitted SVD transformer (150 components) |
| `label_encoder.pkl` | LabelEncoder for 7 classes |
| `X_train_knn.npy` | Dense (840, 158) — LSA + structural |
| `X_train_tfidf.npz` | Sparse (840, 10000) — raw TF-IDF |

---

## Part 3 — Modeling & Evaluation

**Notebook**: `03_Modeling.ipynb`

Two classifiers were trained and evaluated on the held-out test set.

---

### Model Configurations

**KNN:**

| Parameter | Value |
|---|---|
| `n_neighbors` | 5 |
| `metric` | cosine |
| `weights` | distance |
| Feature input | LSA (150d) + Structural (8d) |

**LinearSVC:**

| Parameter | Value |
|---|---|
| `C` | 5 |
| `max_iter` | 3000 |
| Feature input | Sparse TF-IDF (10,000d, bigrams) |

---

### Accuracy Comparison

| Model | Test Accuracy |
|---|---|
| KNN (k=5, cosine) | 83.33% |
| LinearSVC (C=5) | 97.14% |

---

### Confusion Matrices

SVM confusions are sparse and concentrated on the `blog`/`home` boundary. KNN shows broader misclassification, particularly on structurally ambiguous classes.

<img width="2028" height="476" alt="image" src="https://github.com/user-attachments/assets/33c3d9eb-6c0a-4c6c-9601-e29dcbf2b9b6" />

---

### Per-Class F1 Score Comparison

SVM achieves near-perfect F1 on 4 of 7 classes. Both models struggle most with `blog` vs `home`, which share similar vocabulary and structural patterns.

---

### SVM Discriminative Features

The linear decision surface of SVM allows direct inspection of the most discriminative terms per class via coefficient weights.

<img width="2044" height="671" alt="image" src="https://github.com/user-attachments/assets/f1a87dd5-26f5-4c6c-8d98-02bd5a453bf1" />

---

## Final Results

| Model | Accuracy | Macro F1 | Feature Type | Notes |
|---|---|---|---|---|
| KNN (k=5, cosine) | 83.3% | ~0.83 | Dense LSA + structural | Struggles with ambiguous structural classes |
| LinearSVC (C=5) | **97.1%** | **~0.97** | Sparse TF-IDF + bigrams | Near-perfect on 4 of 7 classes |

**Key insights:**
- SVM with TF-IDF dominates because page type is primarily determined by vocabulary
- KNN benefits from structural features but cosine similarity on 150d LSA loses some discriminative signal
- Both models struggle most with `blog` vs `home` - structurally and lexically similar
- All 7 classes are learnable; LinearSVC is the recommended classifier for this task

---

## Tech Stack

| Category | Tools |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| NLP / Features | `scikit-learn` (TfidfVectorizer, TruncatedSVD) |
| Machine Learning | `scikit-learn` (KNeighborsClassifier, LinearSVC) |
| Serialization | `pickle`, `scipy.sparse`, `numpy` |
| Cloud Storage | `AWS S3` |
| Dashboard | `Streamlit` |
| Environment | Python 3.10, Jupyter Notebooks |
