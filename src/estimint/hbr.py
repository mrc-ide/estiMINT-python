"""
Human Biting Rate (HBR) pipeline for estiMINT.

Answers the question: "What happens to EIR if mosquito density increases by X%?"

Pipeline:
1. prev_y9 + interventions -> EIR_baseline       (prevalence model)
2. EIR_baseline + interventions -> HBR_baseline   (EIR-to-HBR model)
3. HBR_new = HBR_baseline * (1 + increase_pct)   (user's mosquito increase)
4. HBR model predicts EIR at both HBR values      (ratio approach)
5. EIR_new = EIR_baseline * (EIR_scaled / EIR_roundtrip)
"""

from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd

from .run import run_xgb_model
from .storage import load_xgb_model


# Lazy-loaded model cache
_models: Dict[str, Any] = {}


def _get_model(name: str) -> Dict[str, Any]:
    """Load and cache a bundled model by name."""
    if name not in _models:
        _models[name] = load_xgb_model(name)
    return _models[name]


def estimate_eir_with_mosquito_increase(
    prevalence: float,
    mosquito_increase: float,
    dn0_use: float,
    Q0: float,
    phi_bednets: float,
    seasonal: float,
    itn_use: float,
    irs_use: float,
    prev_model: Optional[Dict[str, Any]] = None,
    hbr_model: Optional[Dict[str, Any]] = None,
    eir_to_hbr_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Estimate the new EIR after a mosquito density increase.

    Uses a ratio approach: the HBR model predicts EIR at both the baseline
    and scaled HBR, then applies the relative multiplier to the clean
    baseline EIR from the prevalence model.

    Parameters
    ----------
    prevalence : float
        Baseline malaria prevalence (prev_y9), e.g. 0.30 for 30%.
    mosquito_increase : float
        Fractional increase in mosquito density, e.g. 0.10 for +10%.
    dn0_use : float
        Bednet contact reduction parameter.
    Q0 : float
        Human blood index.
    phi_bednets : float
        Proportion of bites on humans while in bed.
    seasonal : float
        Seasonality flag (0.0 or 1.0).
    itn_use : float
        ITN coverage (0-1).
    irs_use : float
        IRS coverage (0-1).
    prev_model : dict, optional
        Custom prevalence model. If None, uses bundled model.
    hbr_model : dict, optional
        Custom HBR->EIR model. If None, uses bundled model.
    eir_to_hbr_model : dict, optional
        Custom EIR->HBR model. If None, uses bundled model.

    Returns
    -------
    dict
        Dictionary with keys:
        - eir_baseline: Baseline EIR from prevalence
        - eir_new: New EIR after mosquito increase
        - eir_multiplier: Ratio of new EIR to baseline
        - hbr_baseline: Estimated baseline HBR
        - hbr_new: HBR after mosquito increase

    Examples
    --------
    >>> from estimint import estimate_eir_with_mosquito_increase
    >>> result = estimate_eir_with_mosquito_increase(
    ...     prevalence=0.30, mosquito_increase=0.25,
    ...     dn0_use=0.33, Q0=0.87, phi_bednets=0.82,
    ...     seasonal=0.0, itn_use=0.6, irs_use=0.0,
    ... )
    >>> print(f"EIR: {result['eir_baseline']:.1f} -> {result['eir_new']:.1f}")
    """
    if prev_model is None:
        prev_model = _get_model("prevalence")
    if hbr_model is None:
        hbr_model = _get_model("hbr")
    if eir_to_hbr_model is None:
        eir_to_hbr_model = _get_model("eir_to_hbr")

    intv = {
        "dn0_use": [dn0_use],
        "Q0": [Q0],
        "phi_bednets": [phi_bednets],
        "seasonal": [seasonal],
        "itn_use": [itn_use],
        "irs_use": [irs_use],
    }

    # Step 1: prevalence -> EIR baseline
    X_prev = pd.DataFrame({"prev_y9": [prevalence], **intv})
    eir_baseline = float(run_xgb_model(X_prev, prev_model)[0])

    if mosquito_increase == 0:
        return {
            "eir_baseline": eir_baseline,
            "eir_new": eir_baseline,
            "eir_multiplier": 1.0,
            "hbr_baseline": 0.0,
            "hbr_new": 0.0,
        }

    # Step 2: EIR -> HBR baseline
    X_eir = pd.DataFrame({"eir": [eir_baseline], **intv})
    hbr_baseline = float(run_xgb_model(X_eir, eir_to_hbr_model)[0])

    # Step 3: apply mosquito increase
    hbr_new = hbr_baseline * (1 + mosquito_increase)

    # Step 4: ratio approach — batch both HBR values in one call so they
    # share the same smooth PCHIP curve
    intv2 = {k: v * 2 for k, v in intv.items()}   # repeat for 2 rows
    X_hbr = pd.DataFrame({"hbr_y9": [hbr_baseline, hbr_new], **intv2})
    eir_both = run_xgb_model(X_hbr, hbr_model)
    eir_rt = float(eir_both[0])
    eir_new_raw = float(eir_both[1])

    # Step 5: multiplier applied to clean baseline
    multiplier = eir_new_raw / eir_rt
    eir_new = eir_baseline * multiplier

    return {
        "eir_baseline": eir_baseline,
        "eir_new": eir_new,
        "eir_multiplier": multiplier,
        "hbr_baseline": hbr_baseline,
        "hbr_new": hbr_new,
    }
