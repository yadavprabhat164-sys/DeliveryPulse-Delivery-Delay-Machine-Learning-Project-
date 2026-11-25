# DeliveryPulse: E-commerce Delivery Delay Prediction

A machine learning project that predicts whether an e-commerce delivery will be **delayed** or arrive **on time**, using information known before the delivery happens (distance, package details, weather, traffic and similar).

> Course project | Built in Python on Google Colab
> Author: *Prabhat Yadav, Aniket Raj, Hiten Awasthi* | Course: *Statistical Machine Learning*  code *CSET211* | Year: 2026

---

## Problem statement

Late deliveries hurt customer trust and raise support costs. If a company can tell in advance which orders are likely to be late, it can warn customers, re-route the package, or assign a faster option. This project builds and compares several classification models that predict that outcome.

- **Type of problem:** binary classification
- **Target:** `is_late` (0 = on time, 1 = delayed)

## Dataset

- **File:** `Delivery_Logistics_55_45.csv`
- **Size:** 25,000 deliveries, 15 columns
- **Class balance:** about 55% on time (13,750) and 45% delayed (11,250)
- **Contents:** delivery partner, package type, vehicle type, delivery mode, region, weather, distance, package weight, expected delivery time and related fields

## Approach

1. **Load and inspect** the data (`head`, `tail`, `shape`, `info`, `describe`).
2. **Clean:** check missing values and duplicates, and repair the `expected_time_hours` column, which was stored as a date-like value.
3. **Create the target:** map `delayed` (`no`/`yes`) to `is_late` (0/1).
4. **Remove data leakage:** drop `delivery_id`, `delayed`, `delivery_status`, `delivery_time_hours` and `delivery_rating`. These either identify the row, contain the answer in another form, or are only known after delivery, so a real model could not use them.
5. **Encode:** one-hot encoding of text columns (`get_dummies`, `drop_first=True`).
6. **Split:** 80% training, 20% testing, stratified (`random_state=42`).
7. **Scale:** `StandardScaler`, fitted on the training data only.
8. **Train and compare** five models and evaluate each with a confusion matrix.

## Models and results

Scores on the held-out test set (5,000 deliveries):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.745 | 0.783 | 0.601 | 0.680 |
| Logistic Regression | 0.747 | 0.764 | 0.633 | 0.692 |
| Linear Regression (threshold 0.5) | 0.749 | 0.771 | 0.628 | 0.692 |
| KNN (K = 5) | 0.660 | 0.634 | 0.579 | 0.605 |
| SVM (RBF kernel) | 0.748 | 0.780 | 0.612 | 0.686 |

**Reading the results**

- Always guessing "on time" would be right about 55% of the time, so about 75% is a clear improvement over guessing.
- Four of the five models land at about 75%, which suggests the limit comes from the information in the data, not the choice of algorithm.
- Precision is higher than recall: when the models say "delayed" they are usually right, but they miss some real delays.
- KNN is weakest because distance-based similarity works poorly with many one-hot columns.

## Evaluation

For every model the notebook reports accuracy, precision, recall, F1 score, a classification report, and a confusion matrix heatmap. A bar chart compares the accuracy of all models.

