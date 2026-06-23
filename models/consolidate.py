"""Build the single estiMINT data source from the two raw simulation DuckDBs.

Joins the epi DB (eir, interventions, prevalence) and the entomology DB (HBR signal)
on (parameter_index, simulation_index) into one year-9 aggregated table. This parquet
is the sole source of truth; the per-model training views are derived from it.

Each column is computed with the same SQL as the original extraction so values are
bit-identical to the legacy pipeline.
"""

import duckdb
import pandas as pd

EPI_DB = "/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
ENTO_DB = "/home/cosmo/Documents/Repos/MINT_DATA/HBR_malaria_simulations_4096.duckdb"
OUT = "/home/cosmo/Documents/Repos/estimint/datasets/estimint_simulations_y9.parquet"

INTERVENTION_DAY = 3285          # day 3285 = 9 years
YEAR9_START = INTERVENTION_DAY - 365

KEYS = ["parameter_index", "simulation_index"]

EPI_QUERY = f"""
SELECT
    parameter_index,
    simulation_index,
    MAX(eir) AS eir,
    MAX(dn0_use) AS dn0_use,
    MAX(Q0) AS Q0,
    MAX(phi_bednets) AS phi_bednets,
    MAX(seasonal) AS seasonal,
    MAX(itn_use) AS itn_use,
    MAX(irs_use) AS irs_use,
    AVG(prevalence) AS prev_y9
FROM simulation_results
WHERE timesteps >= {YEAR9_START} AND timesteps < {INTERVENTION_DAY}
GROUP BY parameter_index, simulation_index
"""

ENTO_QUERY = f"""
SELECT
    parameter_index,
    simulation_index,
    AVG(CASE WHEN Im_Anopheles_count > 0
             THEN EIR_Anopheles * total_M_Anopheles / Im_Anopheles_count END) AS hbr_y9
FROM simulation_results
WHERE timesteps >= {YEAR9_START} AND timesteps < {INTERVENTION_DAY}
GROUP BY parameter_index, simulation_index
"""


def build():
    with duckdb.connect(EPI_DB, read_only=True) as epi:
        df_epi = epi.execute(EPI_QUERY).fetchdf()
    with duckdb.connect(ENTO_DB, read_only=True) as ento:
        df_hbr = ento.execute(ENTO_QUERY).fetchdf()

    df = df_epi.merge(df_hbr, on=KEYS, how="inner").sort_values(KEYS).reset_index(drop=True)

    assert len(df) == 16384, f"expected 16,384 keys, got {len(df):,}"
    df.to_parquet(OUT, index=False)

    print(f"rows={len(df):,}  params={df.parameter_index.nunique()}  sims={df.simulation_index.nunique()}")
    print(f"prev_y9 [{df.prev_y9.min():.4f}, {df.prev_y9.max():.4f}]  "
          f"prev>=0.02: {(df.prev_y9 >= 0.02).sum():,}")
    print(f"hbr_y9  [{df.hbr_y9.min():.2f}, {df.hbr_y9.max():.2f}]  "
          f"hbr>0: {(df.hbr_y9 > 0).sum():,}  null: {df.hbr_y9.isna().sum():,}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
