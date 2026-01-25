"""
Extract training data from MINTelligence DuckDB for estiMINT retraining.

This script:
1. Reads malaria_simulations_4096.duckdb
2. Extracts year 9 prevalence (timestep ~3285) for each parameter set
3. Filters prevalence >= 0.02 (per user policy)
4. Saves as parquet for estiMINT training
"""

import duckdb
import pandas as pd
from pathlib import Path

# Configuration
DUCKDB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
OUTPUT_DIR = Path("/home/cosmo/Documents/Repos/estimint/output/mint_data")
INTERVENTION_DAY = 3285  # Day 3285 = 9 years
MIN_PREVALENCE = 0.02  # User policy: never allow prevalence < 2%
WINDOW = 30  # Days around intervention to consider

def extract_year9_data():
    """Extract year 9 data from all simulations."""
    print(f"Connecting to {DUCKDB_PATH}...")
    conn = duckdb.connect(DUCKDB_PATH)

    # Check data size
    total_rows, total_params = conn.execute("""
        SELECT COUNT(*) as total_rows,
               COUNT(DISTINCT parameter_index) as unique_params
        FROM simulation_results
    """).fetchone()
    print(f"Database contains: {total_rows:,} rows, {total_params:,} parameter sets")

    # Extract year 9 data for all parameter sets
    # We want the closest timestep to INTERVENTION_DAY for each parameter+simulation
    print(f"\nExtracting data near timestep {INTERVENTION_DAY} (year 9)...")

    query = f"""
    WITH ranked AS (
        SELECT
            parameter_index,
            simulation_index,
            eir,
            dn0_use,
            Q0,
            phi_bednets,
            seasonal,
            itn_use,
            irs_use,
            prevalence,
            timesteps,
            ABS(timesteps - {INTERVENTION_DAY}) as time_diff,
            ROW_NUMBER() OVER (
                PARTITION BY parameter_index, simulation_index
                ORDER BY ABS(timesteps - {INTERVENTION_DAY})
            ) as rn
        FROM simulation_results
        WHERE timesteps BETWEEN {INTERVENTION_DAY - WINDOW} AND {INTERVENTION_DAY + WINDOW}
    )
    SELECT
        parameter_index,
        simulation_index,
        eir,
        dn0_use,
        Q0,
        phi_bednets,
        seasonal,
        itn_use,
        irs_use,
        prevalence as prev_y9,
        timesteps
    FROM ranked
    WHERE rn = 1
    """

    print("Running extraction query...")
    df = conn.execute(query).fetchdf()
    conn.close()

    print(f"Extracted {len(df):,} rows (one per parameter×simulation)")

    # Filter by prevalence
    print(f"\nFiltering prevalence >= {MIN_PREVALENCE}...")
    df_filtered = df[df['prev_y9'] >= MIN_PREVALENCE].copy()
    print(f"After filter: {len(df_filtered):,} rows ({len(df) - len(df_filtered):,} removed)")

    # Show statistics
    print("\n=== Data Statistics ===")
    print(f"EIR range: [{df_filtered['eir'].min():.2f}, {df_filtered['eir'].max():.2f}]")
    print(f"Prevalence range: [{df_filtered['prev_y9'].min():.4f}, {df_filtered['prev_y9'].max():.4f}]")
    print(f"dn0 range: [{df_filtered['dn0_use'].min():.4f}, {df_filtered['dn0_use'].max():.4f}]")
    print(f"ITN range: [{df_filtered['itn_use'].min():.4f}, {df_filtered['itn_use'].max():.4f}]")
    print(f"IRS range: [{df_filtered['irs_use'].min():.4f}, {df_filtered['irs_use'].max():.4f}]")
    print(f"Seasonal: {df_filtered['seasonal'].value_counts().to_dict()}")

    # Save as parquet (try) or CSV (fallback)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "mint_training_data.parquet"

    print(f"\nSaving to {output_path}...")
    try:
        df_filtered.to_parquet(output_path, index=False)
        print(f"✓ Done! Training data saved to {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    except ImportError:
        # Fallback to CSV if pyarrow not available
        output_path = OUTPUT_DIR / "mint_training_data.csv"
        print(f"Parquet not available, saving as CSV to {output_path}...")
        df_filtered.to_csv(output_path, index=False)
        print(f"✓ Done! Training data saved to {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path

if __name__ == "__main__":
    extract_year9_data()
