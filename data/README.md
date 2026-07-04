# Data

## Source data before project-specific preprocessing

The official Berkeley Building 59 dataset is available from Dryad:

https://datadryad.org/dataset/doi:10.7941/D1N33Q

The downloaded package contains the separate source CSV groups used before the
project-specific timestamp alignment, gap analysis, interpolation, imputation
and merging stages.

The full source package is not included here because it is large. Reviewers can
download and inspect the original files directly from the official dataset
page.

## Processed data used in the coursework

### `processed/data_merged_data7.csv`

Timestamped merged data used for exploration and electric-load visualisation.

### `processed/merged_using_date_df.csv`

Main model-ready data used for Lasso feature selection and the five-model
regression comparison.

### `processed/mergedfinal_df.csv`

Reduced-feature dataset created during the later cleaning and multicollinearity
stages.

The preprocessing script is stored at:

`code/preprocessing/building59_preprocessing.py`
