# datasets/

Training data for retraining the estiMINT models. Not shipped with the package.

**`estimint_simulations_y9.parquet`** — 16,384 rows (4,096 parameter sets × 4 sims).
`prev_y9` and `hbr_y9` are year-9 means: prevalence and human biting rate averaged over
the 365 days of simulation year 9 (the year ending at the intervention on day 3285).
Columns: `parameter_index`, `simulation_index`, `eir`, `dn0_use`, `Q0`, `phi_bednets`,
`seasonal`, `itn_use`, `irs_use`, `prev_y9`, `hbr_y9`.

Each model's `prepare.py` filters this source and sorts by key into its training view.
