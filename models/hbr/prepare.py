"""Derive the two HBR training views from datasets/estimint_simulations_y9.parquet.

- hbr_training: HBR->EIR model       (hbr_y9 > 0)
- eir_to_hbr_training: EIR->HBR model (prev_y9 >= 0.02 AND hbr_y9 > 0)

Both sorted by key for a deterministic, reproducible view.
"""

import pandas as pd

ROOT = "/home/cosmo/Documents/Repos/estimint"
SOURCE = f"{ROOT}/datasets/estimint_simulations_y9.parquet"

KEYS = ["parameter_index", "simulation_index"]
COLS = KEYS + ["eir", "dn0_use", "Q0", "phi_bednets", "seasonal", "itn_use", "irs_use", "hbr_y9"]
MIN_PREVALENCE = 0.02


def prepare():
    src = pd.read_parquet(SOURCE)

    hbr = src[src.hbr_y9 > 0][COLS].sort_values(KEYS).reset_index(drop=True)
    assert len(hbr) == 16384, f"expected 16,384 hbr rows, got {len(hbr):,}"
    assert not hbr.isna().any().any(), "unexpected NaN in hbr view"
    hbr.to_parquet(f"{ROOT}/models/hbr/hbr_training.parquet", index=False)
    print(f"hbr view:        {len(hbr):,} rows -> models/hbr/hbr_training.parquet")

    e2h = src[(src.prev_y9 >= MIN_PREVALENCE) & (src.hbr_y9 > 0)][COLS].sort_values(KEYS).reset_index(drop=True)
    assert len(e2h) == 12429, f"expected 12,429 eir_to_hbr rows, got {len(e2h):,}"
    assert not e2h.isna().any().any(), "unexpected NaN in eir_to_hbr view"
    e2h.to_parquet(f"{ROOT}/models/hbr/eir_to_hbr_training.parquet", index=False)
    print(f"eir_to_hbr view: {len(e2h):,} rows -> models/hbr/eir_to_hbr_training.parquet")


if __name__ == "__main__":
    prepare()
