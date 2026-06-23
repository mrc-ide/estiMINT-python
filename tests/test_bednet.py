"""Tests for the bednet -> dn0 estimator."""

import pytest

from estimint import calculate_dn0, net_types
from estimint.bednet import DN0Result


class TestNetTypes:
    def test_lists_four_canonical_types(self):
        types = net_types()
        assert types == [
            "pyrethroid_only",
            "pyrethroid_pbo",
            "pyrethroid_ppf",
            "pyrethroid_pyrrole",
        ]


class TestCalculateDn0:
    def test_returns_dn0_result(self):
        res = calculate_dn0(0.5, py_only=1.0)
        assert isinstance(res, DN0Result)
        assert 0.0 <= res.dn0 <= 1.0
        assert res.itn_use == 1.0

    def test_short_and_canonical_names_agree(self):
        assert calculate_dn0(0.5, py_only=1.0) == calculate_dn0(0.5, pyrethroid_only=1.0)

    def test_usage_weighted_average(self):
        a = calculate_dn0(0.4, py_only=1.0).dn0
        b = calculate_dn0(0.4, py_pbo=1.0).dn0
        mixed = calculate_dn0(0.4, py_only=0.5, py_pbo=0.5).dn0
        assert mixed == pytest.approx((a + b) / 2)

    def test_itn_use_sums_pyrethroid_shares(self):
        assert calculate_dn0(0.5, py_only=0.4, py_pbo=0.3, py_ppf=0.1).itn_use == pytest.approx(0.8)

    def test_resistance_lowers_dn0(self):
        # plain pyrethroid nets lose efficacy as resistance rises
        assert calculate_dn0(0.0, py_only=1.0).dn0 > calculate_dn0(1.0, py_only=1.0).dn0

    def test_zero_mix_returns_zero(self):
        assert calculate_dn0(0.5, py_only=0.0) == DN0Result(0.0, 0.0)

    def test_no_nets_raises(self):
        with pytest.raises(ValueError):
            calculate_dn0(0.5)

    def test_unknown_net_raises(self):
        with pytest.raises(ValueError, match="unknown net type"):
            calculate_dn0(0.5, py_supernet=1.0)
