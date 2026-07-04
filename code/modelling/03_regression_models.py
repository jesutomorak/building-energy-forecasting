"""Train and evaluate the regression models used in the coursework."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


DATA_PATH = Path("data/processed/merged_using_date_df.csv")
RESULTS_FOLDER = Path("results")
DATE_COLUMN = "date"
TARGET_COLUMNS = ["mels_S", "lig_S", "mels_N", "hvac_N", "hvac_S"]


def metric_row(model_name: str, target: str, actual, predicted) -> dict:
    return {
        "Model": model_name,
        "Target": target,
        "MSE": mean_squared_error(actual, predicted),
        "MAE": mean_absolute_error(actual, predicted),
        "R2": r2_score(actual, predicted),
    }


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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # Linear Regression without polynomial features
    for target in TARGET_COLUMNS:
        model = LinearRegression()
        model.fit(X_train, y_train[target])
        prediction = model.predict(X_test)
        results.append(
            metric_row("Linear Regression", target, y_test[target], prediction)
        )

    # Linear Regression with degree-2 polynomial features
    polynomial = PolynomialFeatures(degree=2)
    X_train_poly = polynomial.fit_transform(X_train)
    X_test_poly = polynomial.transform(X_test)

    for target in TARGET_COLUMNS:
        model = LinearRegression()
        model.fit(X_train_poly, y_train[target])
        prediction = model.predict(X_test_poly)
        results.append(
            metric_row(
                "Polynomial Linear Regression", target, y_test[target], prediction
            )
        )

    # Random Forest Regression
    for target in TARGET_COLUMNS:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=50,
            random_state=42,
        )
        model.fit(X_train, y_train[target])
        prediction = model.predict(X_test)
        results.append(
            metric_row("Random Forest", target, y_test[target], prediction)
        )

    # Support Vector Regression with standardised predictor variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for target in TARGET_COLUMNS:
        model = SVR(kernel="rbf")
        model.fit(X_train_scaled, y_train[target])
        prediction = model.predict(X_test_scaled)
        results.append(metric_row("SVR", target, y_test[target], prediction))

    # Gradient Boosting Regression
    for target in TARGET_COLUMNS:
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train[target])
        prediction = model.predict(X_test)
        results.append(
            metric_row("Gradient Boosting", target, y_test[target], prediction)
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FOLDER / "model_comparison.csv", index=False)

    for model_name in results_df["Model"].unique():
        print(f"\n{model_name}")
        print(results_df[results_df["Model"] == model_name].to_string(index=False))


if __name__ == "__main__":
    main()
