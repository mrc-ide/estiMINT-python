"""
Extract training data for HBR-based estiMINT model.

This script:
1. Reads HBR_malaria_simulations_4096.duckdb (n_bitten, EIR_Anopheles, total_M, Im)
2. Reads malaria_simulations_4096.duckdb (intervention params + static EIR target)
3. Computes year 9 annual mean HBR = EIR_Anopheles * total_M / Im
4. Joins with intervention parameters
5. Saves as CSV for training
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
HBR_DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/HBR_malaria_simulations_4096.duckdb"
ORIG_DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
OUTPUT_DIR = Path("/home/cosmo/Documents/Repos/estimint/output/hbr_data")
INTERVENTION_DAY = 3285  # Day 3285 = 9 years
YEAR9_START = INTERVENTION_DAY - 365  # Day 2920 = start of year 9


def extract_year9_data():
    """Extract year 9 HBR data from all simulations."""

    # --- Step 1: Extract HBR from HBR database ---
    print(f"Connecting to HBR DB: {HBR_DB_PATH}...")
    conn_hbr = duckdb.connect(HBR_DB_PATH, read_only=True)

    total_rows, total_params = conn_hbr.execute("""
        SELECT COUNT(*) as total_rows,
               COUNT(DISTINCT parameter_index) as unique_params
        FROM simulation_results
    """).fetchone()
    print(f"HBR DB: {total_rows:,} rows, {total_params:,} parameter sets")

    print(f"\nExtracting year 9 mean HBR (days {YEAR9_START}-{INTERVENTION_DAY - 1})...")
    print("HBR = EIR_Anopheles * total_M_Anopheles / Im_Anopheles_count")

    hbr_query = f"""
    SELECT
        parameter_index,
        simulation_index,
        AVG(
            CASE WHEN Im_Anopheles_count > 0
                 THEN EIR_Anopheles * total_M_Anopheles / Im_Anopheles_count
                 ELSE NULL
            END
        ) AS hbr_y9
    FROM simulation_results
    WHERE timesteps >= {YEAR9_START} AND timesteps < {INTERVENTION_DAY}
    GROUP BY parameter_index, simulation_index
    """

    print("Running HBR extraction query...")
    df_hbr = conn_hbr.execute(hbr_query).fetchdf()
    conn_hbr.close()
    print(f"Extracted {len(df_hbr):,} rows from HBR DB")

    # --- Step 2: Extract intervention params from original database ---
    print(f"\nConnecting to original DB: {ORIG_DB_PATH}...")
    conn_orig = duckdb.connect(ORIG_DB_PATH, read_only=True)

    params_query = f"""
    SELECT
        parameter_index,
        simulation_index,
        MAX(eir) AS eir,
        MAX(dn0_use) AS dn0_use,
        MAX(Q0) AS Q0,
        MAX(phi_bednets) AS phi_bednets,
        MAX(seasonal) AS seasonal,
        MAX(itn_use) AS itn_use,
        MAX(irs_use) AS irs_use
    FROM simulation_results
    WHERE timesteps >= {YEAR9_START} AND timesteps < {INTERVENTION_DAY}
    GROUP BY parameter_index, simulation_index
    """

    print("Running intervention params extraction query...")
    df_params = conn_orig.execute(params_query).fetchdf()
    conn_orig.close()
    print(f"Extracted {len(df_params):,} rows from original DB")

    # --- Step 3: Join ---
    print("\nJoining HBR with intervention parameters...")
    df = df_params.merge(df_hbr, on=["parameter_index", "simulation_index"], how="inner")
    print(f"Joined: {len(df):,} rows")

    # --- Step 4: Filter ---
    # Remove rows where HBR is NaN (Im_Anopheles_count was 0 for all timesteps)
    n_before = len(df)
    df = df.dropna(subset=["hbr_y9"])
    n_nan = n_before - len(df)
    if n_nan > 0:
        print(f"Removed {n_nan} rows with NaN HBR (Im=0 throughout year 9)")

    # Remove rows with HBR <= 0
    n_before = len(df)
    df = df[df["hbr_y9"] > 0].copy()
    n_zero = n_before - len(df)
    if n_zero > 0:
        print(f"Removed {n_zero} rows with HBR <= 0")

    print(f"Final dataset: {len(df):,} rows")

    # --- Step 5: Statistics ---
    print("\n=== Data Statistics ===")
    print(f"EIR range: [{df['eir'].min():.2f}, {df['eir'].max():.2f}]")
    print(f"HBR range: [{df['hbr_y9'].min():.2f}, {df['hbr_y9'].max():.2f}]")
    print(f"dn0 range: [{df['dn0_use'].min():.4f}, {df['dn0_use'].max():.4f}]")
    print(f"ITN range: [{df['itn_use'].min():.4f}, {df['itn_use'].max():.4f}]")
    print(f"IRS range: [{df['irs_use'].min():.4f}, {df['irs_use'].max():.4f}]")
    print(f"Seasonal: {df['seasonal'].value_counts().to_dict()}")

    # --- Step 6: Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "hbr_training_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    try:
        parquet_path = OUTPUT_DIR / "hbr_training_data.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Parquet saved to {parquet_path}")
    except ImportError:
        pass

    return csv_path


if __name__ == "__main__":
    extract_year9_data()
