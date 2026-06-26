"""One-call estiMINT -> stateMINT scenario runner (stateMINT imported lazily)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .bednet import calculate_dn0, DN0Result
from .run import run_xgb_model
from .hbr import estimate_eir_with_mosquito_delta
from .storage import load_xgb_model
from .types import Scenario

HF_REPO = "dide-ic/stateMINT"
# 157 windows of 14 days from day 2190; intervention at day 3285.
_ABS_T = 2190 + 14 * np.arange(157)
_IDX_Y9 = int(np.argmin(np.abs(_ABS_T - 3285)))

_MODELS: Dict[str, Any] = {}
_EMULATORS: Dict[str, Dict[str, Any]] = {}

_NET_KEYS = (
    "py_only",
    "py_pbo",
    "py_pyrrole",
    "py_ppf",
)


def _bednet(scenario: Scenario):
    """Current and future net (dn0, itn_use); returns (cur, net_now, net_next)."""
    cur_nets = {net_type: getattr(scenario, net_type) for net_type in _NET_KEYS if getattr(scenario, net_type)}

    net_now = calculate_dn0(scenario.res_use, **cur_nets) if cur_nets else DN0Result(0.0, 0.0)

    net_type_future = scenario.net_type_future
    if scenario.itn_future == 0.0 or not net_type_future:
        net_next = DN0Result(0.0, 0.0)
    else:
        net_next = calculate_dn0(scenario.res_use, **{net_type_future: scenario.itn_future})
    return cur_nets, net_now, net_next


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
                "uv sync --extra scenarios (or pip install "
                '"git+https://github.com/mrc-ide/stateMINT.git@mamba2-train").'
            ) from e
        _EMULATORS[hf_repo] = {
            p: Mamba2Regressor.from_pretrained(hf_repo, predictor=p) for p in ("prevalence", "cases")
        }
    return _EMULATORS[hf_repo]


_EIR_INPUT_COLS = {"prevalence": ("prev_y9", "prevalence"), "hbr": ("hbr_y9", "hbr")}


def _estimate_eir(scenario: Scenario, eir_models: Dict[str, Any]) -> Dict[str, Any]:
    """scenario -> EIR + the stateMINT covariate dict."""
    cur_nets, net_now, net_next = _bednet(scenario)
    dn0_use, itn_use = net_now.dn0, net_now.itn_use
    dn0_future, itn_future = net_next.dn0, net_next.itn_use

    Q0, phi, seasonal = scenario.Q0, scenario.phi, scenario.seasonal
    irs_use = scenario.irs
    irs_future = scenario.irs_future
    routine = scenario.routine
    ppf = scenario.py_ppf
    lsm = scenario.lsm
    if ppf > 0:
        lsm = min(ppf * 0.248 + lsm, 1.0)
    feats = dict(dn0_use=dn0_use, Q0=Q0, phi_bednets=phi, seasonal=seasonal, itn_use=itn_use, irs_use=irs_use)

    # ── EIR by input mode ──
    delta = scenario.mosquito_delta
    hbr_baseline, hbr_new = np.nan, np.nan
    input_mode, input_value = scenario.eir_target.input_mode, scenario.eir_target.input_value

    if input_mode == "eir":
        eir_base = eir_final = input_value
    elif delta and input_mode == "prevalence":
        r = estimate_eir_with_mosquito_delta(prevalence=input_value, mosquito_delta=delta, **feats)
        eir_base, eir_final = r["eir_baseline"], r["eir_new"]
        hbr_baseline, hbr_new = r["hbr_baseline"], r["hbr_new"]
    elif input_mode in _EIR_INPUT_COLS:
        col, model_key = _EIR_INPUT_COLS[input_mode]
        eir_base = eir_final = float(
            run_xgb_model(
                pd.DataFrame({**{k: [v] for k, v in feats.items()}, col: [input_value]}),
                eir_models[model_key],
            )[0]
        )
    else:
        raise ValueError(f"'input' must be 'prevalence', 'hbr' or 'eir'; got {input_mode!r}")

    cov = dict(
        eir=eir_final,
        dn0_use=dn0_use,
        dn0_future=dn0_future,
        Q0=Q0,
        phi_bednets=phi,
        seasonal=seasonal,
        routine=routine,
        itn_use=itn_use,
        irs_use=irs_use,
        itn_future=itn_future,
        irs_future=irs_future,
        lsm=lsm,
    )
    row = dict(
        name=scenario.name,
        input_mode=input_mode,
        net="+".join(cur_nets) or "none",
        net_future=scenario.net_type_future or "none",
        dn0_use=dn0_use,
        itn_use=itn_use,
        irs_use=irs_use,
        dn0_future=dn0_future,
        itn_future=itn_future,
        irs_future=irs_future,
        routine=routine,
        lsm=lsm,
        seasonal=seasonal,
        eir_baseline=eir_base,
        mosquito_delta=delta,
        eir_final=eir_final,
        hbr_baseline=hbr_baseline,
        hbr_new=hbr_new,
    )
    return {"row": row, "cov": cov}


def run_scenarios(
    scenarios: list[Scenario],
    *,
    hf_repo: str = HF_REPO,
) -> pd.DataFrame:
    """Run a list of scenarios through the estiMINT -> stateMINT pipeline."""

    eir_models = _est_models()
    emulator_models = _emulators(hf_repo)

    parts = [_estimate_eir(scenario, eir_models) for scenario in scenarios]
    covs = [p["cov"] for p in parts]
    prev = emulator_models["prevalence"].predict(covs)
    cases = np.maximum(emulator_models["cases"].predict(covs), 0.0)

    return pd.DataFrame(
        [
            {
                **part["row"],
                "prev_y9": float(p[_IDX_Y9]),
                "prev_endline": float(p[-1]),
                "cases_endline": float(c[-1]),
                "prevalence": p,
                "cases": c,
            }
            for part, p, c in zip(parts, prev, cases)
        ]
    )
