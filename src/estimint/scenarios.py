"""One-call estiMINT -> stateMINT scenario runner (stateMINT imported lazily)."""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from .bednet import calculate_dn0
from .run import run_xgb_model
from .hbr import estimate_eir_with_mosquito_delta
from .storage import load_xgb_model

HF_REPO = "dide-ic/stateMINT"
# 157 windows of 14 days from day 2190; intervention at day 3285.
_ABS_T = 2190 + 14 * np.arange(157)
_IDX_Y9 = int(np.argmin(np.abs(_ABS_T - 3285)))

_MODELS: Dict[str, Any] = {}
_EMULATORS: Dict[str, Dict[str, Any]] = {}


def _est_models() -> Dict[str, Any]:
    if not _MODELS:
        _MODELS["prevalence"] = load_xgb_model("prevalence")
        _MODELS["hbr"] = load_xgb_model("hbr")
    return _MODELS


def _emulators(hf_repo: str) -> Dict[str, Any]:
    if hf_repo not in _EMULATORS:
        try:
            from stateMINT.model import Mamba2Regressor
        except ImportError as e:
            raise ImportError(
                "run_scenarios needs stateMINT. Install it with: "
                'uv sync --extra scenarios (or pip install '
                '"git+https://github.com/mrc-ide/stateMINT.git@mamba2-train").'
            ) from e
        _EMULATORS[hf_repo] = {
            p: Mamba2Regressor.from_pretrained(hf_repo, predictor=p)
            for p in ("prevalence", "cases")
        }
    return _EMULATORS[hf_repo]


def _estimate_eir(scn: Dict[str, Any], est: Dict[str, Any]) -> Dict[str, Any]:
    """scenario -> EIR + the stateMINT covariate dict."""
    net = scn.get("net_type_future")
    if net:
        res = calculate_dn0(scn["res_use"], **{net: scn["itn_future"]})
        dn0_use, itn_use = res.dn0, res.itn_use * scn["itn_future"]
    else:
        dn0_use = itn_use = 0.0

    Q0, phi, seasonal = scn["Q0"], scn["phi"], float(scn["seasonal"])
    irs_use, lsm = scn["irs"], scn.get("lsm", 0.0)
    feats = dict(dn0_use=dn0_use, Q0=Q0, phi_bednets=phi,
                 seasonal=seasonal, itn_use=itn_use, irs_use=irs_use)

    mode, value = scn["input"], float(scn["value"])
    if mode == "eir":
        eir_base = value
    elif mode == "prevalence":
        eir_base = float(run_xgb_model(pd.DataFrame({**{k: [v] for k, v in feats.items()}, "prev_y9": [value]}), est["prevalence"])[0])
    elif mode == "hbr":
        eir_base = float(run_xgb_model(pd.DataFrame({**{k: [v] for k, v in feats.items()}, "hbr_y9": [value]}), est["hbr"])[0])
    else:
        raise ValueError(f"'input' must be 'prevalence', 'hbr' or 'eir'; got {mode!r}")

    eir_final, hbr_baseline, hbr_new = eir_base, np.nan, np.nan
    delta = scn.get("mosquito_delta")
    if delta is not None and mode == "prevalence":
        r = estimate_eir_with_mosquito_delta(prevalence=value, mosquito_delta=delta, **feats)
        eir_final, hbr_baseline, hbr_new = r["eir_new"], r["hbr_baseline"], r["hbr_new"]

    # future = use: interventions stay on past day 3285.
    cov = dict(eir=eir_final, dn0_use=dn0_use, dn0_future=dn0_use, Q0=Q0, phi_bednets=phi,
               seasonal=seasonal, routine=0.0, itn_use=itn_use, irs_use=irs_use,
               itn_future=itn_use, irs_future=irs_use, lsm=lsm)
    row = dict(name=scn.get("name"), input_mode=mode, net=net or "none",
               dn0_use=dn0_use, itn_use=itn_use, irs_use=irs_use, lsm=lsm, seasonal=seasonal,
               eir_baseline=eir_base, mosquito_delta=delta, eir_final=eir_final,
               hbr_baseline=hbr_baseline, hbr_new=hbr_new)
    return {"row": row, "cov": cov}


def run_scenarios(
    scenarios: Union[List[Dict[str, Any]], pd.DataFrame],
    *,
    hf_repo: str = HF_REPO,
) -> pd.DataFrame:
    """Run scenarios end-to-end (estiMINT EIR -> stateMINT emulator) -> DataFrame.

    Each scenario: input ("prevalence"|"hbr"|"eir") + value; Q0, phi, seasonal,
    irs; lsm (opt); nets via net_type_future/res_use/itn_future (opt); mosquito_delta (opt,
    prevalence only); name (opt). Output adds prev_y9/prev_endline/cases_endline and the
    length-157 prevalence/cases series.
    """
    if isinstance(scenarios, pd.DataFrame):
        scenarios = scenarios.to_dict(orient="records")
    if not scenarios:
        return pd.DataFrame()

    est = _est_models()
    emu = _emulators(hf_repo)

    parts = [_estimate_eir(scn, est) for scn in scenarios]
    covs = [p["cov"] for p in parts]
    prev = emu["prevalence"].predict(covs)
    cases = np.maximum(emu["cases"].predict(covs), 0.0)

    return pd.DataFrame([
        {**part["row"], "prev_y9": float(p[_IDX_Y9]), "prev_endline": float(p[-1]),
         "cases_endline": float(c[-1]), "prevalence": p, "cases": c}
        for part, p, c in zip(parts, prev, cases)
    ])
