# Regression: Prediction of Experimental Parameters

## Project Overview

This task explores regression techniques to predict four continuous target variables from experimental data. Multiple machine learning algorithms are trained and compared to identify the most suitable model for each target.

## Dataset

- **Dataset:** `fitting-results.csv`
- **Observations:** 19,992
- **Columns:** 9
- **Input features:** 5
- **Target variables:** 4

### Input Features
- `n_cyanos`
- `co2`
- `light`
- `SucRatio`
- `Nsample`

### Target Variables
- `a`
- `mu`
- `tau`
- `a0`

## Methodology

### 1. Exploratory Data Analysis
- Examined feature and target distributions.
- Generated histograms.
- Studied correlations using a correlation heatmap.
- Checked for missing values.

### 2. Data Preprocessing
- Split the data into training (60%), validation (20%), and test (20%) sets.
- Used StandardScaler for Linear Regression and K-Nearest Neighbors.
- Used median imputation for tree-based models.

### 3. Models

Four regression algorithms were compared independently for each target:

- Linear Regression
- K-Nearest Neighbors (KNN) Regressor
- Decision Tree Regressor
- Random Forest Regressor

Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation.

### 4. Evaluation

Models were evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

The best model for each target was selected based on validation RMSE and evaluated on the test set.

## Results

Random Forest Regressor achieved the best validation performance for all four target variables.

| Target | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| a | 2.7088 | 0.5706 | 0.9614 |
| mu | 1.9930 | 0.4928 | 0.9612 |
| tau | 0.0482 | 0.0233 | 0.9869 |
| a0 | 0.0097 | 0.0045 | 0.6887 |

## Discussion and Limitations

Random Forest performed well across most target variables. However, the R² score for `a0` was considerably lower than for the other targets.

This is partly important because `a0` has a very narrow range of values and low variability. Therefore, its evaluation metrics should be interpreted in the context of its distribution.

## Project Structure

- `../notebooks/` – Original notebook containing the classification and regression implementations.
- `results/figures/` – Regression visualizations.
- `results/metrics/` – Regression evaluation results.

## Note

This project is an academic machine learning exercise focused on model comparison and predictive performance, rather than deployment in a production environment.
