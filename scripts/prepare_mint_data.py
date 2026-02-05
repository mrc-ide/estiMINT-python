"""
Extract training data from MINTelligence DuckDB for estiMINT retraining.

This script:
1. Reads malaria_simulations_4096.duckdb
2. Extracts year 9 annual mean prevalence (avg over days 2920-3284) for each parameter set
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
YEAR9_START = INTERVENTION_DAY - 365  # Day 2920 = start of year 9
MIN_PREVALENCE = 0.02  # User policy: never allow prevalence < 2%

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

    # Extract year 9 annual mean prevalence for all parameter sets
    # Average over the full year before intervention (days 2920-3284)
    print(f"\nExtracting annual mean prevalence for year 9 (days {YEAR9_START}-{INTERVENTION_DAY - 1})...")

    query = f"""
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

    # Save as both CSV (for train_on_mint_data.py) and parquet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "mint_training_data.csv"
    df_filtered.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    try:
        parquet_path = OUTPUT_DIR / "mint_training_data.parquet"
        df_filtered.to_parquet(parquet_path, index=False)
        print(f"Parquet saved to {parquet_path}")
    except ImportError:
        pass

    output_path = csv_path

    return output_path

if __name__ == "__main__":
    extract_year9_data()
