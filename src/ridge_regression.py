import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =======================================
# LOGGING CONFIG
# =======================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("./logs/training.log"),  # Log file output
    ],
)

# =======================================
# DATA LOAD
# =======================================

# Load data to pandas dataframe
X_data = pd.read_csv("./data/ENB2012_data.csv")

# Set data types using pandas extension dtypes
X_data = X_data.astype(
    {
        "RelativeCompactness": pd.Float32Dtype(),
        "SurfaceArea": pd.Float32Dtype(),
        "WallArea": pd.Float32Dtype(),
        "RoofArea": pd.Float32Dtype(),
        "OverallHeight": pd.Float32Dtype(),
        "Orientation": pd.CategoricalDtype(),
        "GlazingArea": pd.Float32Dtype(),
        "GlazingAreaDistribution": pd.CategoricalDtype(),
        "HeatingLoad": pd.Float32Dtype(),
        "CoolingLoad": pd.Float32Dtype(),
    }
)

# =======================================
# FEATURE SELECTION FOR PROJECT
# =======================================

# Define the indices of the selected features
select_features_idx = [1, 2, 3, 4, 5, 6]

# Select features and remove any resulting duplicate rows
X_features = X_data.iloc[:, select_features_idx].drop_duplicates()

# Reindex y_data to match the new index of X_features
y_target = X_data.iloc[:, 8].reindex(X_features.index)

# Check if the length of X_features and y_data match after reindexing
assert len(X_features) == len(y_target), "Number of rows do not match after reindexing."

# =======================================
# NESTED CROSS-VALIDATION
# =======================================

# Set up cross-validation
cv_outer = KFold(n_splits=10, shuffle=True)  # Outer loop for generalization performance
cv_inner = KFold(n_splits=10, shuffle=True)  # Inner loop for hyperparameter tuning

# Range of hyperparameters to tune
alphas = np.logspace(-6, 6, 100)  # Generate logarithmically spaced values for alpha

# Create a pipeline with standard scaling of X
# y is not scaled by default and not recommended for ridge
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),  # StandardScaler step
        ("ridge", Ridge()),  # Model step
    ]
)

# Set up parameter grid for GridSearchCV
param_grid = {"ridge__alpha": alphas}

# Set up GridSearchCV (for hyperparameter tuning)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_inner,
    scoring="neg_mean_squared_error",  # Use negative MSE for scoring
    verbose=0,  # Set to 1 to see grid progress
)

# Variables to accumulate RMSEs and alphas from each fold
fold_rmse = []
best_alphas = []

for train_idx, test_idx in cv_outer.split(X_features, y_target):
    # Split data into training and test sets
    X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
    y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]

    # Perform grid search using CV
    grid_search.fit(X_train, y_train)

    # Get the best model and the corresponding hyperparameter
    best_model = grid_search.best_estimator_
    best_alpha = best_model.named_steps["ridge"].alpha
    best_alphas.append(best_alpha)

    # Get predictions from the best model
    y_pred = best_model.predict(X_test)

    # Validation score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    fold_rmse.append(rmse)

    # Log the best hyperparameter and RMSE
    logging.info(f"Best alpha (lambda): {best_alpha:.3f} | RMSE: {rmse:.3f}")

# Calculate and log the approximate generalization error
mean_rmse = np.mean(fold_rmse)
std_rmse = np.std(fold_rmse)
logging.info(
    f"Approximate generalization error (RMSE): {mean_rmse:.3f} | Std: {std_rmse:.3f}"
)

# Calculate and log the average alpha
mean_alpha = np.mean(best_alphas)
logging.info(f"Average alpha for final model: {mean_alpha:.3f}")
