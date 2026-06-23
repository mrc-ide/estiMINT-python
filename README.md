# estiMINT (Python)

Python port of the estiMINT R package for EIR (Entomological Inoculation Rate) estimation using machine learning.

Estimators: EIR from prevalence, EIR↔HBR (incl. the effect of mosquito-density changes), and bednet spec (net type + resistance) → `dn0`.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## File Mapping (R → Python)

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
name (`prevalence`, `hbr`, `eir_to_hbr`) — independent of the training pipeline above.

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

### Bednet → dn0

Map a bednet spec (net-type usage mix + insecticide resistance) to the `dn0`
covariate (probability a mosquito dies on contact), plus total ITN usage:

```python
from estimint import calculate_dn0, net_types

net_types()                      # ['pyrethroid_only', 'pyrethroid_pbo', 'pyrethroid_ppf', 'pyrethroid_pyrrole']
res = calculate_dn0(0.5, py_only=0.4, py_pbo=0.3, py_pyrrole=0.2, py_ppf=0.1)
res.dn0, res.itn_use             # weighted dn0, total net usage
```

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
pip install -e ".[dev]"
pytest
```

Covers the metric/utility helpers plus the three estimator flows: prevalence→EIR
inference, the mosquito-delta HBR pipeline, and bednet→dn0.

## Key Differences from R Version

1. **File format**: Models saved as `.pkl` (pickle) instead of `.rds`
2. **Data handling**: Uses pandas instead of data.table
3. **Plotting**: Uses matplotlib instead of ggplot2
4. **Global model**: Use `set_global_model()` / `get_global_model()` instead of `.GlobalEnv`

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- duckdb >= 0.8.0
- xgboost >= 1.6.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pyarrow >= 10.0.0 (Parquet I/O for the training pipeline)
- requests >= 2.28.0 (optional, for model download)
- appdirs >= 1.4.0 (optional, for cache directory)

## License

MIT License
