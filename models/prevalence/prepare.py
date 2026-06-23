"""Derive the prevalence->EIR training view from the consolidated source.

Filters prev_y9 >= 0.02 and sorts by key for a deterministic, reproducible view.
"""

import pandas as pd

ROOT = "/home/cosmo/Documents/Repos/estimint"
SOURCE = f"{ROOT}/datasets/estimint_simulations_y9.parquet"
OUT = f"{ROOT}/models/prevalence/training.parquet"

KEYS = ["parameter_index", "simulation_index"]
COLS = KEYS + ["eir", "dn0_use", "Q0", "phi_bednets", "seasonal", "itn_use", "irs_use", "prev_y9"]
MIN_PREVALENCE = 0.02


def prepare():
    src = pd.read_parquet(SOURCE)
    view = src[src.prev_y9 >= MIN_PREVALENCE][COLS].sort_values(KEYS).reset_index(drop=True)

    assert len(view) == 12429, f"expected 12,429 rows, got {len(view):,}"
    assert not view.isna().any().any(), "unexpected NaN in prevalence view"
    view.to_parquet(OUT, index=False)
    print(f"prevalence view: {len(view):,} rows -> {OUT}")


if __name__ == "__main__":
    prepare()
