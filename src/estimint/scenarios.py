from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from .bednet import DN0Result, calculate_dn0
from .hbr import estimate_eir_with_mosquito_delta_batch
from .run import run_xgb_model
from .storage import load_xgb_model
from .types import EirTarget, Scenario

####################### Constants and global storage ###################
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

_EST_MODEL_NAMES = ("prevalence", "hbr", "eir_to_hbr")
_EIR_INPUT_COLS = {"prevalence": ("prev_y9", "prevalence"), "hbr": ("hbr_y9", "hbr")}


######################## Internal helpers  ########################
def _load_eir_hbr_models() -> Dict[str, Any]:
    if not _MODELS:
        for name in _EST_MODEL_NAMES:
            _MODELS[name] = load_xgb_model(name)
    return _MODELS


def _load_emulators(hf_repo: str) -> Dict[str, Any]:
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
            predictor: Mamba2Regressor.from_pretrained(hf_repo, predictor=predictor)
            for predictor in ("prevalence", "cases")
        }
    return _EMULATORS[hf_repo]


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


@dataclass
class _ScenarioWork:
    scenario: Scenario
    eir_target: EirTarget
    mosquito_delta: float
    eir_features: Dict[str, float]
    row: dict[str, Any]
    cov: dict[str, float]


def _prepare_scenario(scenario: Scenario) -> _ScenarioWork:
    """Compute non-model scenario values once."""
    cur_nets, net_now, net_next = _bednet(scenario)
    dn0_use, itn_use = net_now.dn0, net_now.itn_use
    dn0_future, itn_future = net_next.dn0, net_next.itn_use

    lsm = scenario.lsm
    if scenario.py_ppf > 0:
        lsm = min(scenario.py_ppf * 0.248 + lsm, 1.0)

    eir_features = {
        "dn0_use": dn0_use,
        "Q0": scenario.Q0,
        "phi_bednets": scenario.phi,
        "seasonal": scenario.seasonal,
        "itn_use": itn_use,
        "irs_use": scenario.irs,
    }

    cov = {
        "eir": np.nan,
        "dn0_use": dn0_use,
        "dn0_future": dn0_future,
        "Q0": scenario.Q0,
        "phi_bednets": scenario.phi,
        "seasonal": scenario.seasonal,
        "routine": scenario.routine,
        "itn_use": itn_use,
        "irs_use": scenario.irs,
        "itn_future": itn_future,
        "irs_future": scenario.irs_future,
        "lsm": lsm,
    }

    row = {
        "name": scenario.name,
        "input_mode": scenario.eir_target.input_mode,
        "net": "+".join(cur_nets) or "none",
        "net_future": scenario.net_type_future or "none",
        "dn0_use": dn0_use,
        "itn_use": itn_use,
        "irs_use": scenario.irs,
        "dn0_future": dn0_future,
        "itn_future": itn_future,
        "irs_future": scenario.irs_future,
        "routine": scenario.routine,
        "lsm": lsm,
        "seasonal": scenario.seasonal,
        "eir_baseline": np.nan,
        "mosquito_delta": scenario.mosquito_delta,
        "eir_final": np.nan,
        "hbr_baseline": np.nan,
        "hbr_new": np.nan,
    }

    return _ScenarioWork(
        scenario=scenario,
        eir_target=scenario.eir_target,
        mosquito_delta=scenario.mosquito_delta,
        eir_features=eir_features,
        row=row,
        cov=cov,
    )


def _set_eir_result(
    work: _ScenarioWork,
    *,
    eir_baseline: float,
    eir_final: float,
    hbr_baseline: float = np.nan,
    hbr_new: float = np.nan,
) -> None:
    work.row["eir_baseline"] = float(eir_baseline)
    work.row["eir_final"] = float(eir_final)
    work.row["hbr_baseline"] = float(hbr_baseline)
    work.row["hbr_new"] = float(hbr_new)
    work.cov["eir"] = float(eir_final)


def _predict_direct_inputs(works: list[_ScenarioWork], *, input_mode: str, eir_models: Dict[str, Any]) -> None:
    """Predict EIR for scenarios with direct inputs (prevalence or hbr)."""
    col, model_key = _EIR_INPUT_COLS[input_mode]
    rows = [{**work.eir_features, col: work.eir_target.input_value} for work in works]

    predictions = run_xgb_model(pd.DataFrame(rows), eir_models[model_key])

    for work, prediction in zip(works, predictions):
        _set_eir_result(work, eir_baseline=prediction, eir_final=prediction)


def _estimate_eir_many(scenarios: list[Scenario], eir_models: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Estimate EIR for many scenarios."""
    if any(scenario.eir_target.input_mode not in {"prevalence", "eir", "hbr"} for scenario in scenarios):
        raise ValueError("All scenarios must have input_mode in {'prevalence', 'eir', 'hbr'}")

    works = [_prepare_scenario(scenario) for scenario in scenarios]

    eir_works = [work for work in works if work.eir_target.input_mode == "eir"]
    for work in eir_works:
        _set_eir_result(work, eir_baseline=work.eir_target.input_value, eir_final=work.eir_target.input_value)

    prevalence_works = [
        work for work in works if work.eir_target.input_mode == "prevalence" and not work.mosquito_delta
    ]
    if prevalence_works:
        _predict_direct_inputs(prevalence_works, input_mode="prevalence", eir_models=eir_models)

    hbr_works = [work for work in works if work.eir_target.input_mode == "hbr"]
    if hbr_works:
        _predict_direct_inputs(hbr_works, input_mode="hbr", eir_models=eir_models)

    mosquito_delta_works = [
        work for work in works if work.eir_target.input_mode == "prevalence" and work.mosquito_delta
    ]
    if mosquito_delta_works:
        mosquito_deltas_inputs = pd.DataFrame(
            [
                {"prevalence": work.eir_target.input_value, "mosquito_delta": work.mosquito_delta, **work.eir_features}
                for work in mosquito_delta_works
            ]
        )
        mosquito_delta_results = estimate_eir_with_mosquito_delta_batch(mosquito_deltas_inputs, models=eir_models)
        for work, result in zip(mosquito_delta_works, mosquito_delta_results.to_dict(orient="records")):
            _set_eir_result(
                work,
                eir_baseline=result["eir_baseline"],
                eir_final=result["eir_new"],
                hbr_baseline=result["hbr_baseline"],
                hbr_new=result["hbr_new"],
            )

    return [{"row": work.row, "cov": work.cov} for work in works]


######################### Public API  ########################
def preload_models(*, hf_repo: str = HF_REPO) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Preload the models used by run_scenarios."""
    eir_models = _load_eir_hbr_models()
    emulator_models = _load_emulators(hf_repo)

    return eir_models, emulator_models


def run_scenarios(
    scenarios: list[Scenario],
    *,
    hf_repo: str = HF_REPO,
) -> pd.DataFrame:
    """Run a list of scenarios through the estiMINT -> stateMINT pipeline."""
    if not scenarios:
        return pd.DataFrame()

    eir_models, emulator_models = preload_models(hf_repo=hf_repo)

    # parts = [_estimate_eir(scenario, eir_models) for scenario in scenarios]
    parts = _estimate_eir_many(scenarios, eir_models)
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
