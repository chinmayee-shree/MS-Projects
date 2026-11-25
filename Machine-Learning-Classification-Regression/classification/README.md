# Classification: Road Accident Injury Severity Prediction

## Project Overview

This task focuses on predicting road accident injury severity using machine learning classification algorithms. The goal is to compare different models and evaluate their ability to classify accident injury severity, particularly in the presence of imbalanced classes.

## Dataset

- **Dataset:** FARS (Fatality Analysis Reporting System)
- **Observations:** 100,968
- **Features:** 29
- **Target variable:** `INJURY_SEVERITY`
- **Classes:** 8 injury severity categories

The dataset contains a mix of numerical and categorical features.

## Methodology

### 1. Data Preprocessing
- Checked the distribution of target classes.
- Used an 80/20 stratified train-test split.
- Applied median imputation to numerical features.
- Applied most-frequent imputation and one-hot encoding to categorical features.
- Used scaling for Logistic Regression.
- Applied SMOTE within a pipeline for the Decision Tree + SMOTE experiment.

### 2. Models

Four classification approaches were compared:

- Decision Tree Classifier
- Decision Tree Classifier with SMOTE
- Random Forest Classifier
- Logistic Regression

Hyperparameter tuning was performed using RandomizedSearchCV with 3-fold stratified cross-validation.

### 3. Evaluation

Models were evaluated using:

- Accuracy
- Balanced Accuracy
- Macro F1-score
- Confusion matrices
- Classification reports

Macro F1-score was used as the cross-validation scoring metric to account for class imbalance.

## Results

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Random Forest | 0.7902 | 0.5552 | 0.5634 |
| Logistic Regression | 0.8037 | 0.5601 | 0.5593 |
| Decision Tree | 0.7915 | 0.5389 | 0.5461 |
| Decision Tree + SMOTE | 0.7862 | 0.5499 | 0.5429 |

Random Forest achieved the highest test Macro F1-score, while Logistic Regression achieved the highest Accuracy and Balanced Accuracy.

## Discussion and Limitations

The results show that predicting injury severity is challenging because of the imbalanced distribution of target classes. Although the models perform reasonably well on more frequent categories, performance on rare categories remains limited.

In particular, the very small number of observations in some classes makes reliable prediction difficult.

## Project Structure

- `../notebooks/` – Original notebook containing the classification and regression implementations.
- `results/figures/` – Classification visualizations and confusion matrices.
- `results/metrics/` – Classification evaluation results.

## Note

This project is an academic machine learning exercise. The models are intended for experimentation and comparison, not real-world accident severity assessment.
