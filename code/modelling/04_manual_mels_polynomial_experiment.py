"""Manual feature experiment for the mels_S target."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


DATA_PATH = Path("data/processed/merged_using_date_df.csv")

SELECTED_FEATURES = [
    "dew_point_temperature_set_1d",
    "solar_radiation_set_1",
    "cerc_templogger_10",
    "cerc_templogger_14",
    "cerc_templogger_2",
    "cerc_templogger_3",
    "cerc_templogger_4",
    "cerc_templogger_5",
    "cerc_templogger_9",
    "rtu_002_sat_sp_tn",
    "rtu_004_fltrd_sa_flow_tn",
    "aru_001_cws_temp",
]
TARGET = "mels_S"


def print_metrics(name: str, actual, predicted) -> None:
    print(name)
    print("Mean Squared Error:", mean_squared_error(actual, predicted))
    print("Mean Absolute Error:", mean_absolute_error(actual, predicted))
    print("R-squared:", r2_score(actual, predicted))
    print()


def main() -> None:
    merged_df = pd.read_csv(DATA_PATH)

    missing_columns = [
        column for column in SELECTED_FEATURES + [TARGET]
        if column not in merged_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = merged_df[SELECTED_FEATURES + [TARGET]].dropna()
    X = data[SELECTED_FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_without_poly = LinearRegression()
    model_without_poly.fit(X_train, y_train)
    prediction_without_poly = model_without_poly.predict(X_test)
    print_metrics(
        "Linear Regression without Polynomial Features",
        y_test,
        prediction_without_poly,
    )

    polynomial = PolynomialFeatures(degree=2)
    X_train_poly = polynomial.fit_transform(X_train)
    X_test_poly = polynomial.transform(X_test)

    model_with_poly = LinearRegression()
    model_with_poly.fit(X_train_poly, y_train)
    prediction_with_poly = model_with_poly.predict(X_test_poly)
    print_metrics(
        "Linear Regression with Polynomial Features",
        y_test,
        prediction_with_poly,
    )


if __name__ == "__main__":
    main()
