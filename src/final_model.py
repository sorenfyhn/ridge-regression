import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import probplot

# =======================================
# LOGGING CONFIG
# =======================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler("./logs/final_model.log"),  # Log file output
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

# Set indices of selected features
select_features_idx = [1, 2, 3, 4, 5, 6]

# Select features and remove any resulting duplicate rows
X_features = X_data.iloc[:, select_features_idx].drop_duplicates()

# Get column names for selected features
feature_names = X_data.columns[select_features_idx].to_list()

# Reindex y_data to match the new index of X_features
y_target = X_data.iloc[:, 8].reindex(X_features.index)

# Check if the length of X_features and y_data match after reindexing
assert len(X_features) == len(y_target), "Number of rows do not match after reindexing."

# =======================================
# FINAL MODEL TRAINING ON FULL DATASET
# =======================================

# Set alpha (lambda)
best_alpha = 0.296

# Create final pipeline with best alpha
final_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha)),
    ]
)

# Fit the final model to the entire dataset
final_model.fit(X_features, y_target)

# =======================================
# RESCALE COEFFICIENTS TO ORIGINAL SCALE
# =======================================

# Extract scaler and ridge model
scaler = final_model.named_steps["scaler"]
ridge = final_model.named_steps["ridge"]

# Get means and scales of the input features
feature_means = scaler.mean_
feature_stds = scaler.scale_

# Rescale coefficients
rescaled_coefs = ridge.coef_ / feature_stds

# Rescale intercept
rescaled_intercept = ridge.intercept_ - np.sum(
    (feature_means / feature_stds) * ridge.coef_
)

# Log descaled coefficients
logging.info("Coefficients (original scale):")
for name, coef in zip(feature_names, rescaled_coefs):
    logging.info(f"{name}: {coef:.3f}")
logging.info(f"Intercept: {rescaled_intercept:.3f}")

# =======================================
# PLOT PREPARATION
# =======================================

# Predict on full dataset
y_pred = final_model.predict(X_features)

# Calculate residuals
residuals = y_target - y_pred

# Standardize the residuals (for QQ plot)
residuals_std = (residuals - np.mean(residuals)) / np.std(residuals)

# Histogram bin size
bin_size = 30

# Default Matplotlib blue for probplot
mat_blue = plt.get_cmap("tab10")(0)

# =======================================
# CREATE A 4x4 PLOT GRID
# =======================================
fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# =======================================
# RESIDUALS
# =======================================
axs[0, 0].scatter(y_pred, residuals, alpha=0.7)
axs[0, 0].axhline(0, color="orange", linestyle="--", linewidth=2)
axs[0, 0].set_title(f"Residuals (alpha={best_alpha})")
axs[0, 0].set_xlabel("Predicted")
axs[0, 0].set_ylabel("Residuals")

# =======================================
# ACTUAL vs PREDICTED
# =======================================
axs[0, 1].scatter(y_target, y_pred, alpha=0.7)
axs[0, 1].plot(
    [y_target.min(), y_target.max()],
    [y_target.min(), y_target.max()],
    color="orange",
    linestyle="--",
    linewidth=2,
)
axs[0, 1].set_title(f"Actual vs Predicted (alpha={best_alpha})")
axs[0, 1].set_xlabel("Actual")
axs[0, 1].set_ylabel("Predicted")

# =======================================
# RESIDUALS DISTRIBUTION
# =======================================
axs[1, 0].hist(residuals, bins=bin_size, edgecolor="black", alpha=0.7)
axs[1, 0].set_title(f"Distribution of Residuals (bins={bin_size})")
axs[1, 0].set_xlabel("Residuals")
axs[1, 0].set_ylabel("Frequency")

# =======================================
# QQ PLOT (STANDARDIZED RESIDUALS)
# =======================================
probplot(residuals_std, dist="norm", plot=axs[1, 1])
axs[1, 1].get_lines()[0].set_color(mat_blue)  # scatter points
axs[1, 1].get_lines()[1].set_color("orange")  # fit line
axs[1, 1].get_lines()[1].set_linestyle("--")
axs[1, 1].set_title("QQ Plot of Standardized Residuals")
axs[1, 1].set_xlabel("Theoretical Quantiles")
axs[1, 1].set_ylabel("Ordered Residuals (Std)")

# =======================================
# FINALIZE AND SAVE PLOT
# =======================================

# Add plot title
fig.suptitle(
    "Final Model Evaluation Summary (Ridge regression)",
    fontsize=14,
    fontweight="bold",
)

# Adjust layout to make room for the supertitle
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as PNG with DPI = 600
plt.savefig("./images/residuals_analysis.png", dpi=600)
