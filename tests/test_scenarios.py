"""Tests for the run_scenarios pipeline.

The estiMINT half (_estimate_eir) is tested offline against the bundled models.
The full run_scenarios call also runs the stateMINT emulator, so it is skipped
unless stateMINT is installed.
"""

import numpy as np
import pytest

from estimint.scenarios import _estimate_eir, _est_models, run_scenarios

INTV = dict(Q0=0.87, phi_bednets=0.82, seasonal=0.0, irs_use=0.0)


@pytest.fixture(scope="module")
def est():
    return _est_models()


class TestEstimateEir:
    def test_prevalence_input(self, est):
        out = _estimate_eir(dict(input="prevalence", value=0.30, **INTV), est)
        assert out["row"]["eir_baseline"] > 0
        assert out["cov"]["eir"] == out["row"]["eir_final"]
        # no nets, no delta
        assert out["row"]["dn0_use"] == 0.0
        assert np.isnan(out["row"]["hbr_baseline"])

    def test_eir_input_passes_through(self, est):
        out = _estimate_eir(dict(input="eir", value=20.0, **INTV), est)
        assert out["row"]["eir_baseline"] == 20.0
        assert out["row"]["eir_final"] == 20.0

    def test_hbr_input(self, est):
        out = _estimate_eir(dict(input="hbr", value=250000.0, **INTV), est)
        assert out["row"]["eir_baseline"] > 0

    def test_bednet_spec_scales_itn_by_usage(self, est):
        # mirrors the demo script: itn_use = calculate_dn0(...).itn_use * net_usage
        out = _estimate_eir(
            dict(input="prevalence", value=0.30, net="pyrethroid_only",
                 resistance=0.30, net_usage=0.70, **INTV), est)
        assert out["row"]["dn0_use"] > 0
        assert out["row"]["itn_use"] == pytest.approx(0.70 * 0.70)

    def test_mosquito_delta_direction(self, est):
        up = _estimate_eir(dict(input="prevalence", value=0.30, mosquito_delta=0.25, **INTV), est)
        down = _estimate_eir(dict(input="prevalence", value=0.30, mosquito_delta=-0.50, **INTV), est)
        assert up["row"]["eir_final"] > up["row"]["eir_baseline"]
        assert down["row"]["eir_final"] < down["row"]["eir_baseline"]
        assert up["row"]["hbr_new"] > up["row"]["hbr_baseline"]

    def test_covariate_dict_keys(self, est):
        out = _estimate_eir(dict(input="eir", value=15.0, lsm=0.3, **INTV), est)
        assert set(out["cov"]) == {
            "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets", "seasonal",
            "routine", "itn_use", "irs_use", "itn_future", "irs_future", "lsm",
        }
        assert out["cov"]["lsm"] == 0.3
        assert out["cov"]["dn0_future"] == out["cov"]["dn0_use"]

    def test_bad_input_raises(self, est):
        with pytest.raises(ValueError, match="input"):
            _estimate_eir(dict(input="nope", value=1.0, **INTV), est)


class TestRunScenariosFullPipeline:
    def test_end_to_end(self):
        pytest.importorskip("stateMINT", reason="stateMINT not installed")
        df = run_scenarios([
            dict(name="prev+delta", input="prevalence", value=0.30,
                 net="pyrethroid_pbo", resistance=0.55, net_usage=0.85,
                 Q0=0.90, phi_bednets=0.85, seasonal=1, irs_use=0.40, mosquito_delta=0.60),
            dict(name="eir", input="eir", value=20.0,
                 Q0=0.88, phi_bednets=0.78, seasonal=1, irs_use=0.60),
        ])
        assert len(df) == 2
        assert {"eir_final", "prev_y9", "prev_series", "cases_series"} <= set(df.columns)
        assert len(df.iloc[0]["prev_series"]) == 157
        assert (df["cases_series"].iloc[0] >= 0).all()
