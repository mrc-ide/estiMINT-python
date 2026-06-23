# estiMINT (Python)

Python port of the estiMINT R package for EIR (Entomological Inoculation Rate) estimation using machine learning.

It estimates EIR from prevalence, converts between EIR and human biting rate (including the effect of changes in mosquito density), and turns a bednet specification (net type and resistance level) into the `dn0` killing parameter.

## Installation

```bash
pip install estimint            # core: inference only (numpy, pandas, xgboost, scipy)
```

Optional extras, by use case:

```bash
pip install "estimint[train]"   # data prep + model training (duckdb, scikit-learn, pyarrow)
pip install "estimint[viz]"     # plotting (matplotlib)
pip install "estimint[all]"     # train + viz + model download
pip install "estimint[dev]"     # test/lint/type-check toolchain
```

The `run_scenarios` pipeline also needs the stateMINT emulator (Python 3.12+). For now it
comes from the `mamba2-train` branch. With uv this is handled for you:

```bash
uv sync --extra scenarios
```

With plain pip, install stateMINT from the branch yourself, then estiMINT:

```bash
pip install "git+https://github.com/mrc-ide/stateMINT.git@mamba2-train"
pip install estimint
```

For local development with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra all --extra dev
```

## File mapping (R to Python)

| R File | Python File | Description |
|--------|-------------|-------------|
| `estiMINT-package.R` | `__init__.py` | Package initialization and exports |
| `globals.R` | `globals.py` | Global variables and constants |
| `utils.R` | `utils.py` | Utility functions (metrics, QMAP, etc.) |
| `data_processing.R` | `data_processing.py` | Data loading and preprocessing |
| `models.R` | `models.py` | XGBoost model training |
| `train.R` | `train.py` | Main training pipeline with K-fold CV |
| `plotting.R` | `plotting.py` | Visualization functions |
| `storage.R` | `storage.py` | Model persistence and loading |
| `run.R` | `run.py` | Model inference |

## Data & retraining pipeline

All training data lives in `datasets/estimint_simulations_y9.parquet`. Two model folders
derive their views from it and train:

```
datasets/               # training data (see datasets/README.md)
models/
  prevalence/           # prev_y9 -> EIR     (estiMINT_model.pkl)
  hbr/                  # HBR<->EIR sub-models (estiMINT_HBR_model.pkl, estiMINT_EIR_to_HBR_model.pkl)
```

Retrain a model end-to-end, e.g. the prevalence model:

```bash
python models/prevalence/prepare.py        # derive the training view from the parquet
python models/prevalence/train.py          # train -> estiMINT_model.pkl + metrics/ + plots/
```

The deployed models shipped with the package live in `src/estimint/data/` and are loaded by
name (`prevalence`, `hbr`, `eir_to_hbr`). This is independent of the training pipeline above.

## API Reference

### Inference

```python
from estimint import load_xgb_model, run_xgb_model
import pandas as pd

# Load a bundled model by name: "prevalence", "hbr", or "eir_to_hbr"
model = load_xgb_model("prevalence")

# Prepare input data
new_data = pd.DataFrame({
    "dn0_use": [0.5],
    "Q0": [0.3],
    "phi_bednets": [0.6],
    "seasonal": [1],
    "itn_use": [0.7],
    "irs_use": [0.2],
    "prev_y9": [0.15]  # or "prevalence"
})

# Run prediction
eir_predictions = run_xgb_model(new_data, model)
print(f"Predicted EIR: {eir_predictions[0]:.2f}")
```

### Using Global Model

```python
from estimint import load_xgb_model, run_xgb_model, set_global_model

# Set global model once
model = load_xgb_model("prevalence")
set_global_model(model)

# Run predictions without passing model
predictions = run_xgb_model(new_data)  # Uses global model
```

### Bednet to dn0

Turn a bednet specification (a mix of net types and an insecticide resistance level) into
the `dn0` covariate, the probability a mosquito dies on contact, along with total ITN usage.

```python
from estimint import calculate_dn0, net_types

net_types()                      # ['pyrethroid_only', 'pyrethroid_pbo', 'pyrethroid_ppf', 'pyrethroid_pyrrole']
res = calculate_dn0(0.5, py_only=0.4, py_pbo=0.3, py_pyrrole=0.2, py_ppf=0.1)
res.dn0, res.itn_use             # weighted dn0, total net usage
```

### Run scenarios

`run_scenarios` runs the whole pipeline in one call. You give it a list of scenarios and
get back a DataFrame. For each scenario it works out the bednet killing effect, estimates
the EIR (from prevalence, from biting rate, or taken directly), optionally adjusts for a
change in mosquito density, then runs the stateMINT emulator forward to the prevalence and
cases trajectories.

This needs the [stateMINT](https://github.com/mrc-ide/stateMINT) package installed as well
as estiMINT. estiMINT only loads it when you call `run_scenarios`, and the model weights
download from HuggingFace.

```python
from estimint import run_scenarios

scenarios = [
    dict(name="PBO nets, prevalence input, 60% more mosquitoes",
         input="prevalence", value=0.30,
         net="pyrethroid_pbo", resistance=0.55, net_usage=0.85,
         Q0=0.90, phi_bednets=0.85, seasonal=1, irs_use=0.40, lsm=0.0,
         mosquito_delta=0.60),
    dict(name="Biting rate input",
         input="hbr", value=250000.0,
         net="pyrethroid_ppf", resistance=0.45, net_usage=0.50,
         Q0=0.80, phi_bednets=0.82, seasonal=0, irs_use=0.0),
    dict(name="EIR supplied directly, no nets",
         input="eir", value=20.0,
         Q0=0.88, phi_bednets=0.78, seasonal=1, irs_use=0.60),
]

df = run_scenarios(scenarios)
print(df[["name", "eir_baseline", "eir_final", "prev_y9", "cases_endline"]])
```

Every scenario needs `input` and `value`, plus `Q0`, `phi_bednets`, `seasonal` and
`irs_use`. `lsm` defaults to 0. To include nets give `net`, `resistance` and `net_usage`,
or leave `net` out for none. `mosquito_delta` only applies when `input` is `"prevalence"`.

The returned DataFrame has one row per scenario. Alongside the inputs it gives the
estimated EIR (`eir_baseline`, and `eir_final` after any mosquito-density change) and the
stateMINT output. That output is year-9 prevalence (`prev_y9`), endline prevalence and
cases, and the full 157-step `prev_series` and `cases_series`. What you do with it is up to
you.

The `estimint.scenarios` module is also where the simulation-based inference and experiment
code will go.

## Utility Functions

```python
from estimint import (
    r2, rmse, mse, mae, median_ae, mae_rel, rmsle, smape,
    fit_qmap_w, predict_qmap_w, scale_pos
)

# Calculate metrics
y_true = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]

print(f"R²: {r2(y_true, y_pred):.4f}")
print(f"RMSE: {rmse(y_true, y_pred):.4f}")
print(f"MAE: {mae(y_true, y_pred):.4f}")

# Quantile mapping calibration
cal = fit_qmap_w(y_pred, y_true)
y_calibrated = predict_qmap_w(y_pred, cal)
```

## Data Processing

These functions need the training extras. Install them with `pip install "estimint[train]"`,
which adds duckdb and scikit-learn.

```python
from estimint import load_and_filter, make_value_weights, strata_and_split

# Load and filter parquet data
result = load_and_filter("data.parquet", thr_lo=0.02, thr_hi=0.95)
df = result["DT"]
df_excluded = result["DT_excluded"]

# Create inverse-frequency weights
weights = make_value_weights(df["eir"].values, digits=3)

# Stratified split
df["eir_log10"] = np.log10(df["eir"])
df = strata_and_split(df, k_strata=16, seed=42)
```

## Testing

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run pytest                # or: pytest
```

This covers the metric and utility helpers, the EIR estimators (prevalence, HBR and direct
EIR), the mosquito-density HBR pipeline, and the bednet calculation.

## CI and releases

The test suite runs on every push and pull request across Python 3.9 to 3.12, defined in
[`.github/workflows/tests.yml`](.github/workflows/tests.yml).

Releases publish to PyPI from [`.github/workflows/publish.yml`](.github/workflows/publish.yml).
It builds with `uv build` and uploads with `uv publish` using
[PyPI trusted publishing](https://docs.astral.sh/uv/guides/integration/github/#publishing-to-pypi),
so no token is stored. To cut a release, bump `version` in `pyproject.toml` and publish a
GitHub Release. The first time, register this repository as a trusted publisher in the PyPI
project settings.

## Key Differences from R Version

1. **File format**: Models saved as `.pkl` (pickle) instead of `.rds`
2. **Data handling**: Uses pandas instead of data.table
3. **Plotting**: Uses matplotlib instead of ggplot2
4. **Global model**: Use `set_global_model()` / `get_global_model()` instead of `.GlobalEnv`

## Dependencies

Core, always installed, and enough for inference:

- numpy >= 1.20.0
- pandas >= 1.3.0
- xgboost >= 1.6.0
- scipy >= 1.7.0

Optional extras, installed with `estimint[name]`:

- `train` adds duckdb, scikit-learn and pyarrow for data prep and model training
- `viz` adds matplotlib for plotting
- `download` adds requests and appdirs for fetching published models
- `all` combines train, viz and download
- `dev` is the test and lint toolchain (pytest, pytest-cov, black, isort, mypy, flake8)

## License

MIT License
