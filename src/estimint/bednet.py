"""Map a bednet spec (net-type usage mix + resistance level) to dn0, via data/itn_dn0.csv."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

_DATA = Path(__file__).parent / "data" / "itn_dn0.csv"

# short alias -> canonical net-type name
_NET_TYPES = {
    "py_only": "pyrethroid_only",
    "py_pbo": "pyrethroid_pbo",
    "py_pyrrole": "pyrethroid_pyrrole",
    "py_ppf": "pyrethroid_ppf",
}
_NET_TYPES.update({v: v for v in _NET_TYPES.values()})


class DN0Result(NamedTuple):
    dn0: float
    itn_use: float


@lru_cache(maxsize=1)
def _splines() -> dict[str, UnivariateSpline]:
    """Interpolating resistance->dn0 spline per net type (loaded/fit once)."""
    df = pd.read_csv(_DATA).sort_values("resistance")
    return {
        nt: UnivariateSpline(g["resistance"].to_numpy(), g["dn0"].to_numpy(), s=0)
        for nt, g in df.groupby("net_type")
    }


def net_types() -> list[str]:
    """Net types available in the dn0 table."""
    return sorted(_splines())


def calculate_dn0(resistance_level: float, **usage: float) -> DN0Result:
    """Usage-weighted dn0 for a net-type mix at a resistance level.

    Net-type shares are keywords, e.g. ``calculate_dn0(0.5, py_only=0.4, py_pbo=0.6)``.
    """
    if not usage:
        raise ValueError("supply at least one <net_type>=<share> pair")

    mix: dict[str, float] = {}
    for name, share in usage.items():
        canon = _NET_TYPES.get(name.lower())
        if canon is None:
            raise ValueError(f"unknown net type: {name!r} (have {sorted(_NET_TYPES)})")
        mix[canon] = mix.get(canon, 0.0) + share

    active = {nt: w for nt, w in mix.items() if w > 0}
    if not active:
        return DN0Result(0.0, 0.0)

    splines = _splines()
    dn0 = float(np.average(
        [float(splines[nt](resistance_level)) for nt in active],
        weights=list(active.values()),
    ))
    itn_use = float(sum(w for nt, w in active.items() if nt.startswith("pyrethroid")))
    return DN0Result(dn0=dn0, itn_use=itn_use)
