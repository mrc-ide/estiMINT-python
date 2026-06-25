"""Tests for the run_scenarios pipeline.

The estiMINT half (_estimate_eir) is tested offline against the bundled models.
The full run_scenarios call also runs the stateMINT emulator, so it is skipped
unless stateMINT is installed.
"""

import numpy as np
import pytest

from estimint.scenarios import Scenario, _estimate_eir, _est_models, run_scenarios

INTV = dict(Q0=0.87, phi=0.82, seasonal=0.0, irs=0.0)


def mk(**kwargs) -> Scenario:
    defaults = dict(name="t", res_use=0.0, **INTV)
    defaults.update(kwargs)
    return Scenario(**defaults)


@pytest.fixture(scope="module")
def est():
    return _est_models()


class TestEstimateEir:
    def test_prevalence_input(self, est):
        out = _estimate_eir(mk(input="prevalence", value=0.30), est)
        assert out["row"]["eir_baseline"] > 0
        assert out["cov"]["eir"] == out["row"]["eir_final"]
        # no nets, no delta
        assert out["row"]["dn0_use"] == 0.0
        assert np.isnan(out["row"]["hbr_baseline"])

    def test_eir_input_passes_through(self, est):
        out = _estimate_eir(mk(input="eir", value=20.0), est)
        assert out["row"]["eir_baseline"] == 20.0
        assert out["row"]["eir_final"] == 20.0

    def test_hbr_input(self, est):
        out = _estimate_eir(mk(input="hbr", value=250000.0), est)
        assert out["row"]["eir_baseline"] > 0

    def test_bednet_mix_matches_minte(self, est):
        # net-type usage mix feeds calculate_dn0 directly, same as minte; itn_use
        # is the sum of pyrethroid shares (NOT rescaled by coverage again).
        out = _estimate_eir(
            mk(input="prevalence", value=0.30, py_only=0.70, py_pbo=0.30, res_use=0.30), est)
        assert out["row"]["dn0_use"] > 0
        assert out["row"]["itn_use"] == pytest.approx(1.00)
        assert out["row"]["net"] == "py_only+py_pbo"

    def test_future_net_switch_is_separate_leg(self, est):
        # net_type_future/itn_future drive the future leg, not current nets;
        # the future leg shares the same res_use as current (no res_future field).
        out = _estimate_eir(
            mk(input="prevalence", value=0.30, py_only=0.50, res_use=0.30,
               net_type_future="pyrethroid_pbo", itn_future=0.70), est)["row"]
        assert out["itn_use"] == pytest.approx(0.50)        # current: py_only=0.50
        assert out["itn_future"] == pytest.approx(0.70)     # future: pbo=0.70
        assert out["dn0_use"] != out["dn0_future"]
        assert out["net"] == "py_only" and out["net_future"] == "pyrethroid_pbo"

    def test_future_without_net_type_is_zeroed(self, est):
        # no net_type_future named -> future leg is zeroed, even if itn_future is set
        # (no carry-forward of the current mix; this is intentional, not a default)
        out = _estimate_eir(
            mk(input="prevalence", value=0.30, py_pbo=0.80, res_use=0.30,
               itn_future=0.70), est)["row"]
        assert out["dn0_future"] == 0.0 and out["itn_future"] == 0.0
        assert out["net_future"] == "none"

    def test_future_nets_removed(self, est):
        # itn_future == 0 removes nets in the future leg
        out = _estimate_eir(
            mk(input="prevalence", value=0.30, py_only=0.60, res_use=0.30,
               itn_future=0.0), est)["row"]
        assert out["itn_use"] == pytest.approx(0.60)
        assert out["dn0_future"] == 0.0 and out["itn_future"] == 0.0

    def test_ppf_boosts_lsm(self, est):
        # PPF nets add larviciding to LSM (minte: py_ppf * 0.248)
        out = _estimate_eir(
            mk(input="eir", value=15.0, py_ppf=0.50, res_use=0.30, lsm=0.10), est)
        assert out["cov"]["lsm"] == pytest.approx(0.50 * 0.248 + 0.10)

    def test_irs_future_and_routine_are_inputs(self, est):
        # irs_future and routine are separate scenario inputs, like minte
        out = _estimate_eir(
            mk(input="eir", value=15.0, irs=0.40, irs_future=0.10, routine=0.25), est)["cov"]
        assert out["irs_use"] == 0.40 and out["irs_future"] == 0.10
        assert out["routine"] == 0.25

    def test_irs_future_and_routine_defaults(self, est):
        # irs_future is a static dataclass default (0.0); it does NOT carry irs
        # forward automatically, even when irs is nonzero.
        out = _estimate_eir(mk(input="eir", value=15.0, irs=0.40), est)["cov"]
        assert out["irs_future"] == 0.0
        assert out["irs_use"] == 0.40
        assert out["routine"] == 0.0

    def test_mosquito_delta_direction(self, est):
        up = _estimate_eir(mk(input="prevalence", value=0.30, mosquito_delta=0.25), est)
        down = _estimate_eir(mk(input="prevalence", value=0.30, mosquito_delta=-0.50), est)
        assert up["row"]["eir_final"] > up["row"]["eir_baseline"]
        assert down["row"]["eir_final"] < down["row"]["eir_baseline"]
        assert up["row"]["hbr_new"] > up["row"]["hbr_baseline"]

    def test_covariate_dict_keys(self, est):
        out = _estimate_eir(mk(input="eir", value=15.0, lsm=0.3), est)
        assert set(out["cov"]) == {
            "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets", "seasonal",
            "routine", "itn_use", "irs_use", "itn_future", "irs_future", "lsm",
        }
        assert out["cov"]["lsm"] == 0.3
        assert out["cov"]["dn0_future"] == out["cov"]["dn0_use"]

    def test_bad_input_raises(self, est):
        with pytest.raises(ValueError, match="input"):
            _estimate_eir(mk(input="nope", value=1.0), est)


class TestRunScenariosFullPipeline:
    def test_end_to_end(self):
        pytest.importorskip("stateMINT", reason="stateMINT not installed")
        df = run_scenarios([
            Scenario(name="prev+delta", input="prevalence", value=0.30,
                     py_only=0.60, res_use=0.55, net_type_future="pyrethroid_pbo",
                     itn_future=0.85,
                     Q0=0.90, phi=0.85, seasonal=1, irs=0.40, mosquito_delta=0.60),
            Scenario(name="eir", input="eir", value=20.0, res_use=0.0,
                     Q0=0.88, phi=0.78, seasonal=1, irs=0.60),
        ])
        assert len(df) == 2
        assert {"eir_final", "prev_y9", "prevalence", "cases"} <= set(df.columns)
        assert len(df.iloc[0]["prevalence"]) == 157
        assert (df["cases"].iloc[0] >= 0).all()
