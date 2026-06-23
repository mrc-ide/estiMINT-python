# datasets/

Training data for retraining the estiMINT models. Not shipped with the package.

**`estimint_simulations_y9.parquet`** — 16,384 rows (4,096 parameter sets × 4 sims),
year-9 aggregates. Columns: `parameter_index`, `simulation_index`, `eir`, `dn0_use`,
`Q0`, `phi_bednets`, `seasonal`, `itn_use`, `irs_use`, `prev_y9`, `hbr_y9`.

Rebuild from the raw DuckDBs in `MINT_DATA/`:
```
python models/consolidate.py
```

Each model's `prepare.py` filters this source and sorts by key into its training view.
