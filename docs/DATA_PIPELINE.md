# Data pipeline

```text
Official Building 59 source CSV groups
        ↓
Per-folder preprocessing script
        ↓
Continuous 15-minute timestamps
        ↓
Gap measurement and short-gap interpolation
        ↓
KNN imputation
        ↓
Matrix factorisation for very long gaps
        ↓
data_<filename>.csv and parameter_<filename>.csv
        ↓
Processed files merged by date
        ↓
data_merged_data7.csv
        ↓
merged_using_date_df.csv / mergedfinal_df.csv
        ↓
Exploration, VIF and Lasso feature selection
        ↓
Regression-model comparison
```

After per-file preprocessing, the processed data groups were merged by date to create the modelling datasets used in the analysis.

