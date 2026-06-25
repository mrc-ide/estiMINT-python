"""End-to-end flow tests: prevalence -> EIR, and the mosquito-delta HBR pipeline.

These exercise the bundled models in src/estimint/data, so they run offline.
"""

import pandas as pd

from estimint import (
    load_xgb_model,
    run_xgb_model,
    estimate_eir_with_mosquito_delta,
)

INTERVENTIONS = dict(
    dn0_use=0.33, Q0=0.87, phi_bednets=0.82,
    seasonal=0.0, itn_use=0.6, irs_use=0.0,
)


class TestPrevalenceToEir:
    def test_predicts_positive_eir(self):
        model = load_xgb_model("prevalence")
        X = pd.DataFrame({"prev_y9": [0.30], **{k: [v] for k, v in INTERVENTIONS.items()}})
        eir = run_xgb_model(X, model)
        assert len(eir) == 1
        assert eir[0] > 0

    def test_higher_prevalence_gives_higher_eir(self):
        model = load_xgb_model("prevalence")
        rows = {"prev_y9": [0.10, 0.50], **{k: [v, v] for k, v in INTERVENTIONS.items()}}
        eir = run_xgb_model(pd.DataFrame(rows), model)
        assert eir[1] > eir[0]


class TestMosquitoDelta:
    def _run(self, delta):
        return estimate_eir_with_mosquito_delta(
            prevalence=0.30, mosquito_delta=delta, **INTERVENTIONS
        )

    def test_returns_expected_keys(self):
        res = self._run(0.25)
        assert set(res) == {
            "eir_baseline", "eir_new", "eir_multiplier", "hbr_baseline", "hbr_new",
        }

    def test_zero_delta_is_identity(self):
        res = self._run(0.0)
        assert res["eir_new"] == res["eir_baseline"]
        assert res["eir_multiplier"] == 1.0

    def test_more_mosquitoes_raises_eir(self):
        res = self._run(0.25)
        assert res["eir_new"] > res["eir_baseline"]
        assert res["eir_multiplier"] > 1.0
        assert res["hbr_new"] > res["hbr_baseline"]

    def test_fewer_mosquitoes_lowers_eir(self):
        res = self._run(-0.50)
        assert res["eir_new"] < res["eir_baseline"]
        assert res["hbr_new"] < res["hbr_baseline"]

    def test_monotonic_in_delta(self):
        eirs = [self._run(d)["eir_new"] for d in (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)]
        assert eirs == sorted(eirs)
