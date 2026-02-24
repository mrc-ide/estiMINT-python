"""
Prepare training data for EIR -> HBR model.

This script:
1. Loads mint_training_data.csv (has eir, interventions, prev_y9)
2. Loads hbr_training_data.csv (has eir, interventions, hbr_y9)
3. Joins on (parameter_index, simulation_index)
4. Outputs training data for eir + interventions -> hbr_y9 model
"""

import pandas as pd
from pathlib import Path

# Configuration
PREV_DATA = "/home/cosmo/Documents/Repos/estimint/output/mint_data/mint_training_data.csv"
HBR_DATA = "/home/cosmo/Documents/Repos/estimint/output/hbr_data/hbr_training_data.csv"
OUTPUT_DIR = Path("/home/cosmo/Documents/Repos/estimint/output/eir_to_hbr_data")


def prepare_data():
    print("Loading prevalence training data...")
    df_prev = pd.read_csv(PREV_DATA)
    print(f"  {len(df_prev):,} rows")

    print("Loading HBR training data...")
    df_hbr = pd.read_csv(HBR_DATA)
    print(f"  {len(df_hbr):,} rows")

    # Join on (parameter_index, simulation_index)
    print("\nJoining on (parameter_index, simulation_index)...")
    df = df_prev.merge(
        df_hbr[["parameter_index", "simulation_index", "hbr_y9"]],
        on=["parameter_index", "simulation_index"],
        how="inner"
    )
    print(f"Joined: {len(df):,} rows")

    # Keep: eir, interventions, hbr_y9 (target)
    cols = ["parameter_index", "simulation_index",
            "eir", "dn0_use", "Q0", "phi_bednets", "seasonal",
            "itn_use", "irs_use", "hbr_y9"]
    df = df[cols].copy()

    # Stats
    print("\n=== Data Statistics ===")
    print(f"EIR range: [{df['eir'].min():.2f}, {df['eir'].max():.2f}]")
    print(f"HBR range: [{df['hbr_y9'].min():.2f}, {df['hbr_y9'].max():.2f}]")
    print(f"Rows: {len(df):,}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "eir_to_hbr_training_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path}")

    return csv_path


if __name__ == "__main__":
    prepare_data()
