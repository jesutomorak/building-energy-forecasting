"""Lasso feature selection for the five electric-load targets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/merged_using_date_df.csv")
RESULTS_FOLDER = Path("results")
DATE_COLUMN = "date"
TARGET_COLUMNS = ["mels_S", "lig_S", "mels_N", "hvac_N", "hvac_S"]


def main() -> None:
    merged_df = pd.read_csv(DATA_PATH)
    RESULTS_FOLDER.mkdir(exist_ok=True)

    if DATE_COLUMN in merged_df.columns:
        merged_df[DATE_COLUMN] = pd.to_datetime(
            merged_df[DATE_COLUMN], errors="coerce"
        )
        merged_df["day_of_week"] = merged_df[DATE_COLUMN].dt.dayofweek
        merged_df["month"] = merged_df[DATE_COLUMN].dt.month
        merged_df["quarter"] = merged_df[DATE_COLUMN].dt.quarter
        merged_df["year"] = merged_df[DATE_COLUMN].dt.year

    X = merged_df.drop(
        columns=TARGET_COLUMNS + [DATE_COLUMN], errors="ignore"
    ).select_dtypes(include=np.number)
    y = merged_df[TARGET_COLUMNS]

    combined = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(how="all")
    imputer = KNNImputer(n_neighbors=5)
    imputed = pd.DataFrame(
        imputer.fit_transform(combined),
        columns=combined.columns,
        index=combined.index,
    )
    X = imputed[X.columns]
    y = imputed[TARGET_COLUMNS]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    selected_rows = []
    for target_column in TARGET_COLUMNS:
        lasso_model = Lasso(alpha=0.1)
        lasso_model.fit(X_train_scaled, y_train[target_column])

        selected = X_train.columns[np.abs(lasso_model.coef_) > 0]
        print(f"Selected features for {target_column}:")
        print(list(selected))
        print()

        for feature, coefficient in zip(X_train.columns, lasso_model.coef_):
            if coefficient != 0:
                selected_rows.append(
                    {
                        "Target": target_column,
                        "Feature": feature,
                        "Coefficient": coefficient,
                    }
                )

    pd.DataFrame(selected_rows).to_csv(
        RESULTS_FOLDER / "lasso_selected_features.csv", index=False
    )


if __name__ == "__main__":
    main()
