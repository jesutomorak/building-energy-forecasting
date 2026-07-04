# Energy Consumption Forecasting in Buildings

**Author:** Jesutomito Morakinyo  
**Project type:** Academic machine-learning coursework  
**Dataset:** Berkeley Building 59 operational data

## Project overview

This project develops and compares regression models for predicting five
building electric-load variables from weather, indoor-environment and HVAC
operational measurements.

The repository is organised to show:

1. the source dataset before my project-specific preprocessing;
2. the preprocessing code used during the coursework;
3. the processed datasets produced by that workflow;
4. the feature-selection, multicollinearity and regression analysis;
5. the saved outputs from the completed hold-out evaluation.

## Source data before project preprocessing

The source files can be downloaded from the official Dryad dataset page:

**[A three-year building operational performance dataset for informing energy efficiency](https://datadryad.org/dataset/doi:10.7941/D1N33Q)**

The source package contains separate CSV groups for electricity, weather,
temperature, HVAC operation, indoor conditions and occupancy-related data.

The large source package is not duplicated in this repository. The processed
datasets used for the coursework modelling are included under
`data/processed/`.

## Preprocessing code

The preprocessing script is available at:

```text
code/preprocessing/building59_preprocessing.py
```

It documents the coursework preprocessing workflow. The path-specific
logic and methods used in the project are retained.

The workflow:

- parsed and sorted timestamps;
- created continuous 15-minute time indices;
- measured missing-data gaps and negative-value rates;
- used linear interpolation for gaps of one hour or less;
- applied KNN imputation with three neighbours;
- used matrix factorisation for gaps of at least one day;
- saved `data_<filename>.csv` and `parameter_<filename>.csv` outputs.

The script was applied separately to selected Building 59 data folders. It is
included for documentation and is not required to rerun the forecasting models.

## Target variables

| Variable | Description |
|---|---|
| `mels_S` | Miscellaneous electric loads in the south section |
| `lig_S` | Lighting electricity load in the south section |
| `mels_N` | Miscellaneous electric loads in the north section |
| `hvac_N` | HVAC electricity load in the north section |
| `hvac_S` | HVAC electricity load in the south section |

MELS refers to miscellaneous electrical loads such as computers, office
equipment and other plug-connected devices. Each target is modelled separately.

## Repository structure

```text
building-energy-forecasting/
├── code/
│   ├── preprocessing/
│   │   └── building59_preprocessing.py
│   └── modelling/
│       ├── 01_exploration_correlation_vif.py
│       ├── 02_feature_selection_lasso.py
│       ├── 03_regression_models.py
│       └── 04_manual_mels_polynomial_experiment.py
├── notebooks/
│   └── Energy_Consumption_Forecasting_Executed.ipynb
├── data/
│   ├── processed/
│   │   ├── data_merged_data7.csv
│   │   ├── merged_using_date_df.csv
│   │   └── mergedfinal_df.csv
│   └── README.md
├── docs/
│   ├── DATA_PIPELINE.md
│   └── PROJECT_RESULTS.md
├── results/
│   ├── correlation_heatmap.png
│   ├── electric_loads_over_time.png
│   ├── correlation_matrix.csv
│   ├── vif_results.csv
│   ├── lasso_selected_features.csv
│   └── model_comparison.csv
├── requirements.txt
├── requirements-preprocessing.txt
├── .gitignore
└── README.md
```

## Processed datasets

### `data_merged_data7.csv`

Timestamped merged data used for exploration, missing-value inspection,
correlation analysis, VIF calculation and electric-load visualisation.

### `merged_using_date_df.csv`

The main model-ready dataset used for Lasso feature selection and regression
comparison. It contains the five target variables and the calendar features
used during modelling.

### `mergedfinal_df.csv`

A reduced-feature dataset created during the cleaning and
multicollinearity stages.

See [`data/README.md`](data/README.md) for further details.

## Models

The completed comparison includes:

- Linear Regression;
- degree-2 Polynomial Linear Regression;
- Random Forest Regression;
- Support Vector Regression;
- Gradient Boosting Regression.

The reported results use an 80/20 random train-test split with
`random_state=42` and are evaluated with MSE, MAE and R².

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The preprocessing script additionally uses `fancyimpute`. Install
it only if you intend to adapt and rerun that path-specific script:

```bash
pip install -r requirements-preprocessing.txt
```

## Running the modelling analysis

Run commands from the repository root:

```bash
python code/modelling/01_exploration_correlation_vif.py
python code/modelling/02_feature_selection_lasso.py
python code/modelling/03_regression_models.py
python code/modelling/04_manual_mels_polynomial_experiment.py
```

These scripts write their outputs to `results/`.

The saved executed notebook can be opened with:

```bash
jupyter notebook notebooks/Energy_Consumption_Forecasting_Executed.ipynb
```

## Saved hold-out results

Random Forest produced the strongest overall performance across the five
targets.

| Target | Random Forest R² |
|---|---:|
| `mels_S` | 0.971 |
| `lig_S` | 0.791 |
| `mels_N` | 0.963 |
| `hvac_N` | 0.922 |
| `hvac_S` | 0.945 |

The complete saved comparison is available in
[`results/model_comparison.csv`](results/model_comparison.csv).

## Scope and limitations

The model results use a random hold-out split rather than a chronological
future-data split. The project is therefore best interpreted as
regression-based building-energy prediction using historical operational data,
rather than strict future time-series forecasting.

After per-file preprocessing, the processed data groups were merged by date
to create the retained modelling datasets included in this repository.
