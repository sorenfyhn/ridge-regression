
# Ridge regression

This repository contains my generic framework for prototyping a ridge regression model. It is based on project work I did for a Machine Learning course at DTU. It is a fully working example with the included dataset, but it is also easy to adapt for a new project.

## Structure of the scripts

### ridge_regression.py

- Load data into a pandas DataFrame.
- Set pandas extension dtypes to optimize memory usage and support nullability.
- Manually select features, remove duplicate rows, and reindex the DataFrame.
- Tune `alpha` using the inner loop of nested cross-validation.
- Validation and training errors (RMSE) for each fold are logged to `training.log` for evaluating overfitting.
- Estimate generalization error (RMSE) from the outer loop and log it to `training.log`.
- Compute average `best_alpha` across folds and log it to `training.log`.

### final_model.py

- Load data into a pandas DataFrame.
- Set pandas extension dtypes to optimize memory usage and support nullability.
- Manually select features, remove duplicate rows, and reindex the DataFrame.
- Set `best_alpha` and train the final model on the full dataset.
- Rescale model coefficients and log them to `final_model.log`.
- Compute residuals and save a diagnostic plot to `residuals_analysis.png`.

![Diagnostic plot](/images/residuals_analysis.png)

## Dataset

Tsanas, Athanasios and Angeliki Xifara. 2012. Energy Efficiency. UCI Machine Learning Repository. <https://doi.org/10.24432/C51307>.
