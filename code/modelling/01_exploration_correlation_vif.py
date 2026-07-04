"""Dataset exploration, correlation analysis and VIF calculation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


DATA_PATH = Path("data/processed/data_merged_data7.csv")
RESULTS_FOLDER = Path("results")
DATE_COLUMN = "date"
TARGET_COLUMNS = ["mels_S", "lig_S", "mels_N", "hvac_N", "hvac_S"]


def main() -> None:
    merged_df = pd.read_csv(DATA_PATH)
    RESULTS_FOLDER.mkdir(exist_ok=True)

    print("Dataset shape:", merged_df.shape)
    print("\nData types:")
    print(merged_df.dtypes)
    print("\nSummary statistics:")
    print(merged_df.describe(include="all").T)
    print("\nMissing values:")
    print(merged_df.isna().sum().sort_values(ascending=False).head(30))

    if DATE_COLUMN in merged_df.columns:
        merged_df[DATE_COLUMN] = pd.to_datetime(
            merged_df[DATE_COLUMN], errors="coerce"
        )
        merged_df = merged_df.sort_values(DATE_COLUMN)

        plt.figure(figsize=(14, 7))
        for target in TARGET_COLUMNS:
            plt.plot(merged_df[DATE_COLUMN], merged_df[target], label=target)
        plt.xlabel("Date")
        plt.ylabel("Electric load")
        plt.title("Building Electric Loads Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_FOLDER / "electric_loads_over_time.png", dpi=200)
        plt.close()

    numeric_df = merged_df.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()
    correlation_matrix.to_csv(RESULTS_FOLDER / "correlation_matrix.csv")

    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / "correlation_heatmap.png", dpi=200)
    plt.close()

    feature_df = merged_df.drop(
        columns=TARGET_COLUMNS + [DATE_COLUMN], errors="ignore"
    ).select_dtypes(include=np.number)
    feature_df = feature_df.loc[:, feature_df.nunique(dropna=False) > 1]

    imputer = KNNImputer(n_neighbors=5)
    feature_values = imputer.fit_transform(feature_df)

    vif_data = pd.DataFrame(
        {
            "Feature": feature_df.columns,
            "VIF": [
                variance_inflation_factor(feature_values, index)
                for index in range(feature_values.shape[1])
            ],
        }
    ).sort_values("VIF", ascending=False)

    vif_data.to_csv(RESULTS_FOLDER / "vif_results.csv", index=False)
    print("\nVariance Inflation Factors:")
    print(vif_data.to_string(index=False))


if __name__ == "__main__":
    main()
