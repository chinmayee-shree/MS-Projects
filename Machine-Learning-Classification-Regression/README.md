# Machine Learning: Classification and Regression

## Project Overview

This repository contains two machine learning tasks developed as part of a machine learning coursework project:

1. **Classification:** Prediction of road accident injury severity using the Fatality Analysis Reporting System (FARS) dataset.
2. **Regression:** Prediction of experimental parameters using regression algorithms.

Both tasks explore data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and performance evaluation.

## Project 1: Classification

The classification task aims to predict road accident injury severity across eight categories.

**Dataset:** FARS (Fatality Analysis Reporting System)

**Models implemented:**
- Decision Tree Classifier
- Decision Tree Classifier with SMOTE
- Random Forest Classifier
- Logistic Regression

**Evaluation metrics:**
- Accuracy
- Balanced Accuracy
- Macro F1-score

Random Forest achieved the highest test Macro F1-score, while Logistic Regression achieved the highest Accuracy and Balanced Accuracy.

[View Classification Details](classification/README.md)

## Project 2: Regression

The regression task predicts four continuous target variables (`a`, `mu`, `tau`, and `a0`) using five input features.

**Dataset:** Experimental fitting results (`fitting-results.csv`)

**Models implemented:**
- Linear Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Regressor
- Random Forest Regressor

**Evaluation metrics:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

Random Forest achieved the best validation performance for all four target variables.

[View Regression Details](regression/README.md)

## Repository Structure

```text
machine-learning-coursework/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── machine_learning_coursework.ipynb
│
├── classification/
│   ├── README.md
│   └── results/
│       ├── figures/
│       └── metrics/
│
└── regression/
    ├── README.md
    └── results/
        ├── figures/
        └── metrics/
```

## Tools and Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Google Colab

## How to Use

1. Clone or download this repository.
2. Install the required Python libraries using:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook in Google Colab or Jupyter Notebook.
4. Update the dataset paths to match your environment.
5. Run the notebook cells to reproduce the analysis and model evaluations.

## Limitations

The classification task is affected by class imbalance, particularly for rare injury severity categories. The regression task also includes a target variable (`a0`) with low variability, which affects the interpretation of its R² score.

The results represent academic experimentation and model comparison, not production-ready predictive systems.

## Author

Chinmayee Shree N R
