# models/hbr

The HBR feature's two sub-models, both used by `estimate_eir_with_mosquito_delta`
(`src/estimint/hbr.py`) to answer "what happens to EIR if mosquito density changes by X%?".

| Sub-model | Direction | Bundle name | File |
|---|---|---|---|
| `train_hbr_to_eir.py` | HBR + interventions → EIR | `hbr` | `estiMINT_HBR_model.pkl` |
| `train_eir_to_hbr.py` | EIR + interventions → HBR | `eir_to_hbr` | `estiMINT_EIR_to_HBR_model.pkl` |

```bash
python models/hbr/prepare.py            # source -> hbr_training.parquet + eir_to_hbr_training.parquet
python models/hbr/train_hbr_to_eir.py   # -> estiMINT_HBR_model.pkl
python models/hbr/train_eir_to_hbr.py   # -> estiMINT_EIR_to_HBR_model.pkl
```

Deployed copies live in `src/estimint/data/`.
