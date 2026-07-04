"""Building 59 preprocessing workflow used in the coursework.

The script was applied separately to selected folders of Building 59 CSV
files. It creates a continuous 15-minute time index, measures data gaps,
uses linear interpolation for short gaps, applies KNN imputation, and uses
matrix factorisation for gaps longer than one day.

The Windows input and output paths reflect the coursework environment and
must be changed before running on another computer.
"""

import csv
import datetime
import math
import os
import time

import numpy as np
import pandas as pd
from fancyimpute import KNN, MatrixFactorization
from pandas import Series
from sklearn.impute import KNNImputer


# Windows path used in the coursework environment.
# The input and output folders were changed for each Building 59 data group.
path = (
    r"C:\Users\presi\Downloads\Electricity_Prediction_data"
    r"\doi_10_7941_D1N33Q__v20220202\Building_59\zone_temp_exterior"
)
files = os.listdir(path)

path_postprocess = (
    r"C:\Users\presi\Downloads\Electricity_Prediction_data"
    r"\doi_10_7941_D1N33Q__v20220202\Building_59\site_weather_postprocess"
)


# Read data files and adjust time format.
for filename in files:
    row = pd.read_csv(path + "\\" + filename)
    row["date"] = pd.to_datetime(row["date"], infer_datetime_format=True)

    helper = pd.DataFrame(
        {
            "date": pd.date_range(
                row["date"].min(),
                row["date"].max(),
                freq="15min",
            )
        }
    )

    row = pd.merge(row, helper, on="date", how="outer").sort_values("date")

    count_out = Series([0], index=["date"])       # Count of outlier values
    count_gap = Series([0], index=["date"])       # Count of gaps
    count_outgap = Series([0], index=["date"])    # Count of large gaps
    gap_max = Series([0], index=["date"])         # Maximum gap

    # Calculate the number of gaps and interpolate according to gap size.
    for i in range(1, len(row.columns)):
        k = 0
        out_gapcount = 0
        start_index = {}
        starttime = {}
        end_index = {}
        endtime = {}
        gap = {}

        if (
            pd.isnull(row.iloc[len(row.index) - 1, i]) is True
            or math.isnan(row.iloc[len(row.index) - 1, i]) is True
        ):
            row.iloc[len(row.index) - 1, i] = 0

        for j in range(0, len(row.index)):
            if (
                (pd.isnull(row.iloc[j, i]) or math.isnan(row.iloc[j, i]))
                and pd.isnull(row.iloc[j - 1, i]) is False
            ):
                starttime[k] = row.iloc[j - 1, 0]
                start_index[k] = j - 1

            elif (
                (pd.isnull(row.iloc[j - 1, i]) or math.isnan(row.iloc[j - 1, i]))
                and pd.isnull(row.iloc[j, i]) is False
            ):
                endtime[k] = row.iloc[j, 0]
                end_index[k] = j
                k = k + 1

        if k != 0:
            for m in range(k):
                starttime_struct = datetime.datetime.strptime(
                    str(starttime[m]),
                    "%Y-%m-%d %H:%M:%S",
                )
                endtime_struct = datetime.datetime.strptime(
                    str(endtime[m]),
                    "%Y-%m-%d %H:%M:%S",
                )
                gap[m] = (endtime_struct - starttime_struct).total_seconds()

                if gap[m] <= 3600:
                    # Linear interpolation when the gap is one hour or less.
                    if len(row.iloc[start_index[m] : end_index[m] + 1, i]) > 0:
                        row.iloc[
                            start_index[m] : end_index[m] + 1,
                            i,
                        ] = row.iloc[
                            start_index[m] : end_index[m] + 1,
                            i,
                        ].interpolate(method="linear")

                elif gap[m] > 3600 * 24:
                    out_gapcount = out_gapcount + 1

            maxgap = max(gap.values()) / 60
            gap_max = pd.concat(
                [gap_max, pd.Series(maxgap, index=[row.columns[i]])]
            )

        outcount = np.sum(row.iloc[:, i] < 0) / len(row)

        count_out = pd.concat(
            [count_out, pd.Series(outcount, index=[row.columns[i]])]
        )
        count_gap = pd.concat(
            [count_gap, pd.Series(k, index=[row.columns[i]])]
        )
        count_outgap = pd.concat(
            [
                count_outgap,
                pd.Series(out_gapcount, index=[row.columns[i]]),
            ]
        )

    row_interpolation = np.array(row.iloc[:, 1:], dtype=np.float64)

    imputer = KNNImputer(n_neighbors=3)
    row_interpolation = imputer.fit_transform(row_interpolation)

    # Restore gaps of at least one day to missing before matrix factorisation.
    for i in range(1, len(row.columns)):
        k = 0
        start_index = {}
        starttime = {}
        end_index = {}
        endtime = {}

        for j in range(0, len(row.index)):
            if (
                pd.isnull(row.iloc[j, i])
                and pd.isnull(row.iloc[j - 1, i]) is False
            ):
                starttime[k] = row.iloc[j - 1, 0]
                start_index[k] = j - 1

            elif (
                pd.isnull(row.iloc[j - 1, i])
                and pd.isnull(row.iloc[j, i]) is False
            ):
                endtime[k] = row.iloc[j, 0]
                end_index[k] = j
                k = k + 1

        for m in range(k):
            starttime_struct = datetime.datetime.strptime(
                str(starttime[m]),
                "%Y-%m-%d %H:%M:%S",
            )
            endtime_struct = datetime.datetime.strptime(
                str(endtime[m]),
                "%Y-%m-%d %H:%M:%S",
            )
            gap[m] = (endtime_struct - starttime_struct).total_seconds()

            if gap[m] >= 3600 * 24:
                row_interpolation[
                    start_index[m] : end_index[m] + 1,
                    i - 1,
                ] = None

    if out_gapcount != 0:
        row_interpolation = MatrixFactorization().fit_transform(
            row_interpolation
        )

    row.iloc[:, 1:] = row_interpolation

    cols_not_null = (len(row) - row.count(axis=0)) / len(row)

    data = pd.DataFrame(
        {
            "missingrate": cols_not_null,
            "outrate": count_out,
            "count_outgap": count_outgap,
            "count_gap": count_gap,
            "maxgap": gap_max,
        }
    )

    data.to_csv(
        path_postprocess + "\\" + "parameter_" + filename,
        sep=",",
        header=True,
        index=True,
    )

    row.to_csv(
        path_postprocess + "\\" + "data_" + filename,
        sep=",",
        header=True,
        index=False,
    )
