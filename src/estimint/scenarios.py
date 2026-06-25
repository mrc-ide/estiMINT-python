"""One-call estiMINT -> stateMINT scenario runner (stateMINT imported lazily)."""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from .bednet import calculate_dn0, DN0Result
from .run import run_xgb_model
from .hbr import estimate_eir_with_mosquito_delta
from .storage import load_xgb_model

HF_REPO = "dide-ic/stateMINT"
# 157 windows of 14 days from day 2190; intervention at day 3285.
_ABS_T = 2190 + 14 * np.arange(157)
_IDX_Y9 = int(np.argmin(np.abs(_ABS_T - 3285)))

_MODELS: Dict[str, Any] = {}
_EMULATORS: Dict[str, Dict[str, Any]] = {}

_NET_KEYS = ("py_only", "py_pbo", "py_pyrrole", "py_ppf",
             "pyrethroid_only", "pyrethroid_pbo", "pyrethroid_pyrrole", "pyrethroid_ppf")


def _bednet(scn: Dict[str, Any]):
    """Current and future net (dn0, itn_use); returns (cur, net_now, net_next)."""
    cur = {nt: float(scn[nt]) for nt in _NET_KEYS if scn.get(nt)}
    res_use = float(scn.get("res_use", 0.0))
    res_future = float(scn["res_future"]) if scn.get("res_future") is not None else res_use

    net_now = calculate_dn0(res_use, **cur) if cur else DN0Result(0.0, 0.0)

    net_future = scn.get("net_type_future")
    itn_future = scn.get("itn_future")
    itn_future = None if itn_future is None else float(itn_future)
    if itn_future == 0.0:
        net_next = DN0Result(0.0, 0.0)
    elif not net_future or itn_future is None:
        net_next = calculate_dn0(res_future, **cur) if cur else DN0Result(0.0, 0.0)
    else:
        net_next = calculate_dn0(res_future, **{net_future: itn_future})
    return cur, net_now, net_next


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
    cur, net_now, net_next = _bednet(scn)
    dn0_use, itn_use = net_now.dn0, net_now.itn_use
    dn0_future, itn_future = net_next.dn0, net_next.itn_use

    Q0, phi, seasonal = scn["Q0"], scn["phi"], float(scn["seasonal"])
    irs_use = scn["irs"]
    irs_future = scn.get("irs_future", irs_use)
    routine = scn.get("routine", 0.0)
    ppf = float(scn.get("py_ppf", 0.0)) + float(scn.get("pyrethroid_ppf", 0.0))
    lsm = float(scn.get("lsm", 0.0))
    if ppf > 0:
        lsm = min(ppf * 0.248 + lsm, 1.0)
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

    cov = dict(eir=eir_final, dn0_use=dn0_use, dn0_future=dn0_future, Q0=Q0, phi_bednets=phi,
               seasonal=seasonal, routine=routine, itn_use=itn_use, irs_use=irs_use,
               itn_future=itn_future, irs_future=irs_future, lsm=lsm)
    row = dict(name=scn.get("name"), input_mode=mode, net="+".join(cur) or "none",
               net_future=scn.get("net_type_future") or "none",
               dn0_use=dn0_use, itn_use=itn_use, irs_use=irs_use,
               dn0_future=dn0_future, itn_future=itn_future, irs_future=irs_future,
               routine=routine, lsm=lsm, seasonal=seasonal,
               eir_baseline=eir_base, mosquito_delta=delta, eir_final=eir_final,
               hbr_baseline=hbr_baseline, hbr_new=hbr_new)
    return {"row": row, "cov": cov}


def run_scenarios(
    scenarios: Union[List[Dict[str, Any]], pd.DataFrame],
    *,
    hf_repo: str = HF_REPO,
) -> pd.DataFrame:
    
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
