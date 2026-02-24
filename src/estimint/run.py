"""
Model inference functions for estiMINT package.

Equivalent to: run.R
"""

from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import xgboost as xgb

from .utils import predict_qmap_w


# Global model storage (equivalent to R's .GlobalEnv)
_global_model: Optional[Dict[str, Any]] = None


def set_global_model(model: Dict[str, Any]) -> None:
    """
    Set the global estiMINT model.
    
    Parameters
    ----------
    model : dict
        An 'estiMINT_model' object
    """
    global _global_model
    _global_model = model


def get_global_model() -> Optional[Dict[str, Any]]:
    """
    Get the global estiMINT model.
    
    Returns
    -------
    dict or None
        The global model if set, None otherwise
    """
    return _global_model


def run_xgb_model(
    new_data: Union[pd.DataFrame, Dict[str, Any]],
    model: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Run XGBoost model with initial conditions.
    
    Equivalent to R's run_xgb_model() function.
    
    Parameters
    ----------
    new_data : pd.DataFrame or dict
        Data frame with columns: prevalence (or prev_y9), dn0_use, Q0, 
        phi_bednets, seasonal, itn_use, irs_use
    model : dict, optional
        An 'estiMINT_model' object; if None, tries global 'estiMINT_model'
        
    Returns
    -------
    np.ndarray
        Numeric array of calibrated EIR predictions
        
    Raises
    ------
    ValueError
        If no model is provided and no global model exists
    ValueError
        If required columns are missing from new_data
    """
    # Get model
    if model is None:
        model = get_global_model()
        if model is None:
            raise ValueError(
                "No model provided and 'estiMINT_model' not found in the global context. "
                "Either pass a model or call set_global_model() first."
            )
    
    # Get required features
    req = model["features"]
    
    # Convert to DataFrame if necessary
    if isinstance(new_data, dict):
        nd = pd.DataFrame(new_data)
    else:
        nd = new_data.copy()
    
    # Handle prevalence -> prev_y9 alias
    if "prevalence" in nd.columns and "prev_y9" not in nd.columns:
        nd["prev_y9"] = nd["prevalence"]
    
    # Check for missing columns
    missing = set(req) - set(nd.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    
    # Extract feature matrix
    X = nd[req].values.astype(np.float64)

    # Detect monotonic feature.  For models with a monotonic input
    # (prev_y9 or hbr_y9), we always predict via a dense internal sweep
    # + PCHIP smoothing so that every query — even a single point —
    # returns a value from a smooth, monotone curve rather than the raw
    # XGBoost staircase.
    # Fixed sweep ranges.  prev_y9 is linear; hbr_y9 is log-spaced because
    # HBR spans four orders of magnitude in the training data (8k–57M).
    _MONO_SWEEP = {
        "prev_y9": {"lo": 0.005, "hi": 0.80, "log": False},
        "hbr_y9":  {"lo": 5000,  "hi": 70_000_000, "log": True},
    }
    mono_fidx = None
    mono_cfg = None
    for fname, cfg in _MONO_SWEEP.items():
        if fname in req:
            mono_fidx = req.index(fname)
            mono_cfg = cfg
            break

    if mono_fidx is None:
        # No monotonic feature (e.g. EIR-to-HBR model) — predict directly
        return _predict_direct(X, model)

    # Group rows by unique intervention combo (all columns except the
    # monotonic feature) so each group gets its own smooth curve.
    other_cols = [i for i in range(X.shape[1]) if i != mono_fidx]
    other_vals = X[:, other_cols]

    if len(X) == 1:
        unique_combos = other_vals
        combo_labels = np.array([0])
    else:
        unique_combos, combo_labels = np.unique(
            other_vals, axis=0, return_inverse=True
        )

    pred_final = np.empty(len(X), dtype=np.float64)
    n_sweep = 501

    for ci in range(len(unique_combos)):
        mask = combo_labels == ci
        query_fvals = X[mask, mono_fidx]
        intv = unique_combos[ci]

        # Dense sweep with fixed range (same curve for every call)
        if mono_cfg["log"]:
            sweep_fvals = np.logspace(
                np.log10(mono_cfg["lo"]), np.log10(mono_cfg["hi"]), n_sweep
            )
        else:
            sweep_fvals = np.linspace(mono_cfg["lo"], mono_cfg["hi"], n_sweep)

        # Build the sweep feature matrix (insert mono column at correct index)
        X_sweep = np.tile(intv, (n_sweep, 1))
        X_sweep = np.insert(X_sweep, mono_fidx, sweep_fvals, axis=1)

        # Predict sweep → staircase → PCHIP smooth curve
        sweep_preds = _predict_direct(X_sweep, model)
        sweep_smooth = _smooth_staircase(sweep_fvals, sweep_preds)

        # Interpolate query points from the smooth curve
        pred_final[mask] = np.interp(query_fvals, sweep_fvals, sweep_smooth)

    return pred_final.astype(np.float64)


def _predict_direct(X: np.ndarray, model: dict) -> np.ndarray:
    """Raw model prediction + QMAP calibration (no smoothing)."""
    dnew = xgb.DMatrix(X)
    pred_log10 = model["booster"].predict(dnew)
    pred_raw = np.power(10, pred_log10)
    pred_cal = predict_qmap_w(pred_raw, model["calibrator"]["qmap"])
    return np.maximum(0, model["calibrator"]["scale"] * pred_cal).astype(np.float64)


def _smooth_staircase(fvals: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Smooth a monotone staircase into a visually smooth monotone curve.

    Two-stage smoothing:
    1. PCHIP through staircase midpoint knots — removes hard edges, gives
       C1-continuous monotone curve.
    2. Gaussian kernel smoothing — spreads steep PCHIP transitions over a
       wider range so that 1% user steps always show gradual change, never
       "nothing then big jump".
    3. Re-enforce strict monotonicity after Gaussian pass.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d

    order = np.argsort(fvals)
    x = fvals[order]
    y = preds[order]

    # Find constant segments (within relative tolerance)
    segments = []
    i = 0
    while i < len(y):
        j = i + 1
        ref = max(abs(y[i]), 1e-10)
        while j < len(y) and abs(y[j] - y[i]) / ref < 1e-6:
            j += 1
        segments.append((i, j, y[i]))
        i = j

    if len(segments) <= 2:
        return preds  # too few segments to interpolate

    # Knot at the midpoint of each segment's feature range
    knots_x = [(x[s[0]] + x[s[1] - 1]) / 2 for s in segments]
    knots_y = [s[2] for s in segments]

    # Pin the boundary knots to the actual data range so PCHIP covers
    # the full sweep without extrapolation
    knots_x[0] = x[0]
    knots_x[-1] = x[-1]

    knots_x = np.array(knots_x)
    knots_y = np.array(knots_y)

    # Stage 1: PCHIP — smooth monotone cubic interpolation
    pchip = PchipInterpolator(knots_x, knots_y)
    y_smooth = pchip(x)

    # Stage 2: Gaussian smoothing in log-space to spread steep transitions.
    # Work in log-space so the kernel acts on relative (multiplicative)
    # changes rather than absolute — this prevents the Gaussian from
    # under-smoothing at the low end and over-smoothing at the high end.
    # sigma=12 on a 501-point sweep ≈ 2.4% of the range, enough to spread
    # a sharp step across ~5% of the sweep (several user-facing 1% steps).
    y_log = np.log(np.maximum(y_smooth, 1e-10))
    y_log_smooth = gaussian_filter1d(y_log, sigma=12, mode="nearest")
    y_smooth = np.exp(y_log_smooth)

    # Stage 3: enforce strict monotonicity (Gaussian can create tiny dips)
    for i in range(1, len(y_smooth)):
        if y_smooth[i] <= y_smooth[i - 1]:
            y_smooth[i] = y_smooth[i - 1] * (1 + 1e-10)

    # Restore original order
    result = np.empty_like(preds)
    result[order] = y_smooth
    return result
