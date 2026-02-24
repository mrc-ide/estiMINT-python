"""
Train estiMINT EIR-to-HBR model.

This script:
1. Loads the prepared data (eir + interventions -> hbr_y9)
2. Trains XGBoost model using K-fold CV and QMAP calibration
3. Saves model to output/eir_to_hbr_retrained/

This is the "reverse" model: given a baseline EIR and interventions,
predict what the human biting rate is. Used to derive baseline HBR
so users can apply percentage changes (e.g. "10% more mosquitoes").
"""

import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.cluster import KMeans

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from estimint.utils import (
    ts, r2, rmse, mse, mae, median_ae, mae_rel, rmsle,
    safe_div, smape, fit_qmap_w, predict_qmap_w, scale_pos
)
from estimint.data_processing import make_value_weights
from estimint.plotting import plot_obs_pred

# Configuration
DATA_PATH = "/home/cosmo/Documents/Repos/estimint/output/eir_to_hbr_data/eir_to_hbr_training_data.csv"
OUTPUT_DIR = "/home/cosmo/Documents/Repos/estimint/output/eir_to_hbr_retrained"
K_FOLDS = 10
K_STRATA = 16
SEED = 42

def main():
    print("=" * 80)
    print("Training estiMINT EIR-to-HBR Model (eir + interventions -> hbr_y9)")
    print("=" * 80)

    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print(f"ERROR: Training data not found at {DATA_PATH}")
        print("Please run prepare_eir_to_hbr_data.py first")
        return 1

    print(f"\nInput data: {data_path}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Create output directories
    out_dir = Path(OUTPUT_DIR)
    dir_models = out_dir / "models"
    dir_plots = out_dir / "plots"
    dir_metric = out_dir / "metrics"
    dir_pred = out_dir / "predictions"

    for d in [dir_models, dir_plots, dir_metric, dir_pred]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    ts("Loading training data...")
    DT = pd.read_csv(data_path)
    print(f"Loaded {len(DT):,} rows")

    # Features: eir + interventions. Target: hbr_y9
    features = ["eir", "dn0_use", "Q0", "phi_bednets", "seasonal", "itn_use", "irs_use"]

    # Transform HBR to log10 (large range, same reason as EIR)
    DT["hbr_log10"] = np.log10(DT["hbr_y9"])

    # XGBoost parameters
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "seed": SEED,
    }

    # Create strata using k-means on log10(HBR)
    ts("Creating %d strata on log10(HBR) and 70/15/15 split...", K_STRATA)
    np.random.seed(SEED)

    hbr_log10 = DT["hbr_log10"].values.reshape(-1, 1)
    km = KMeans(n_clusters=K_STRATA, n_init=50, max_iter=5000, random_state=SEED)
    km.fit(hbr_log10)

    centers = km.cluster_centers_.flatten()
    ord_idx = np.argsort(centers)
    id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(ord_idx)}
    DT["strat_bin"] = np.array([id_map[c] for c in km.labels_])

    # Stratified split
    DT["split"] = None
    for b in sorted(DT["strat_bin"].unique()):
        idx = DT[DT["strat_bin"] == b].index.tolist()
        n_b = len(idx)
        n_tr = int(np.floor(0.70 * n_b))
        n_val = int(np.floor(0.15 * n_b))

        np.random.shuffle(idx)

        tr_idx = idx[:n_tr] if n_tr > 0 else []
        val_idx = idx[n_tr:n_tr + n_val] if n_val > 0 else []
        te_idx = idx[n_tr + n_val:]

        DT.loc[tr_idx, "split"] = "train"
        DT.loc[val_idx, "split"] = "val"
        DT.loc[te_idx, "split"] = "test"

    DT["split"] = DT["split"].fillna("train")

    # Hold-out test set
    DT_test = DT[DT["split"] == "test"]
    X_test = DT_test[features].values.astype(np.float64)
    y_test = DT_test["hbr_log10"].values
    obs_hbr_test = np.power(10, y_test)

    ts("Test set: %d rows", len(DT_test))

    # CV folds
    ts("Assigning %d-fold CV within TRAIN+VAL strata...", K_FOLDS)
    DTcv = DT[DT["split"] != "test"].copy()

    np.random.seed(SEED + 1)

    DTcv["fold"] = 0
    for b in DTcv["strat_bin"].unique():
        mask = DTcv["strat_bin"] == b
        n_b = mask.sum()
        idx = DTcv.index[mask].tolist()
        np.random.shuffle(idx)
        folds = np.tile(np.arange(1, K_FOLDS + 1), int(np.ceil(n_b / K_FOLDS)))[:n_b]
        np.random.shuffle(folds)
        DTcv.loc[idx, "fold"] = folds

    # K-fold CV training
    ts("Running %d-fold CV with early stopping...", K_FOLDS)
    oof_pred_raw = np.full(len(DTcv), np.nan)
    best_iters = np.zeros(K_FOLDS, dtype=int)

    for k in range(1, K_FOLDS + 1):
        ts(" Fold %d / %d", k, K_FOLDS)

        idx_val = DTcv["fold"] == k
        idx_tr = DTcv["fold"] != k

        X_tr = DTcv.loc[idx_tr, features].values.astype(np.float64)
        y_tr = DTcv.loc[idx_tr, "hbr_log10"].values
        X_va = DTcv.loc[idx_val, features].values.astype(np.float64)
        y_va = DTcv.loc[idx_val, "hbr_log10"].values

        w_tr = make_value_weights(np.power(10, y_tr), digits=3)
        w_va = make_value_weights(np.power(10, y_va), digits=3)

        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dva = xgb.DMatrix(X_va, label=y_va, weight=w_va)

        mdl = xgb.train(
            params=xgb_params,
            dtrain=dtr,
            num_boost_round=5000,
            evals=[(dtr, "train"), (dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        best_iters[k - 1] = mdl.best_iteration
        pred_log10_va = mdl.predict(dva)
        oof_pred_raw[idx_val.values] = np.power(10, pred_log10_va)

    obs_cv_raw = np.power(10, DTcv["hbr_log10"].values)

    # Fit calibrator
    ts("Fitting final calibrator (QMAP + positive scale) on OOF...")
    cal_oof = fit_qmap_w(oof_pred_raw, obs_cv_raw, ngrid=1024, round_digits=8)
    oof_pred_cal = predict_qmap_w(oof_pred_raw, cal_oof)
    a_oof = scale_pos(obs_cv_raw, oof_pred_cal)
    oof_pred_final = np.maximum(0, a_oof * oof_pred_cal)

    # OOF metrics
    oof_metrics = pd.DataFrame({
        "set": ["OOF_uncalibrated", "OOF_calibrated"],
        "R2": [r2(obs_cv_raw, oof_pred_raw), r2(obs_cv_raw, oof_pred_final)],
        "bias": [np.mean(oof_pred_raw - obs_cv_raw), np.mean(oof_pred_final - obs_cv_raw)],
        "RMSE": [rmse(obs_cv_raw, oof_pred_raw), rmse(obs_cv_raw, oof_pred_final)],
        "MAE": [mae(obs_cv_raw, oof_pred_raw), mae(obs_cv_raw, oof_pred_final)],
    })
    oof_metrics.to_csv(dir_metric / f"hbr_OOF_metrics_K{K_FOLDS}CV.csv", index=False)
    print("\n" + str(oof_metrics))

    # Train final model
    ts("Training final model on TRAIN+VAL with nrounds = median(best_iteration)...")
    best_nrounds = int(np.round(np.median(best_iters)))
    print(f"Best nrounds: {best_nrounds}")

    DT_trcv = DT[DT["split"] != "test"]
    X_trcv = DT_trcv[features].values.astype(np.float64)
    y_trcv = DT_trcv["hbr_log10"].values
    w_trcv = make_value_weights(np.power(10, y_trcv), digits=3)

    dtrcv = xgb.DMatrix(X_trcv, label=y_trcv, weight=w_trcv)

    xgb_final = xgb.train(
        params=xgb_params,
        dtrain=dtrcv,
        num_boost_round=best_nrounds,
        verbose_eval=False,
    )
    xgb_final.save_model(str(dir_models / "hbr_xgb_FINAL.model"))

    # Predict on TEST
    dtest = xgb.DMatrix(X_test, label=y_test)
    pred_log10_test_raw = xgb_final.predict(dtest)
    pred_raw_test = np.power(10, pred_log10_test_raw)
    pred_hbr_test = predict_qmap_w(pred_raw_test, cal_oof)
    pred_hbr_test = np.maximum(0, a_oof * pred_hbr_test)

    # Test metrics
    test_metrics = pd.DataFrame({
        "set": ["Test"],
        "R2": [r2(obs_hbr_test, pred_hbr_test)],
        "bias": [np.mean(pred_hbr_test - obs_hbr_test)],
        "RMSE": [rmse(obs_hbr_test, pred_hbr_test)],
        "MAE": [mae(obs_hbr_test, pred_hbr_test)],
    })
    test_metrics.to_csv(dir_metric / "hbr_test_metrics.csv", index=False)
    print("\n" + str(test_metrics))

    # Plot
    plot_obs_pred(
        obs_hbr_test, pred_hbr_test,
        f"HBR — Observed vs Predicted (XGBoost, K={K_FOLDS} CV, QMAP+Scale, test)",
        str(dir_plots / "hbr_obs_vs_pred_xgb_QMAP_SCALE_test.png"),
        xlab="Observed HBR", ylab="Predicted HBR"
    )

    # Model bundle
    cal_bundle = {
        "kind": "qmap+scale",
        "qmap": {"xq": cal_oof["xq"], "yq": cal_oof["yq"]},
        "scale": a_oof
    }

    preprocess = {
        "features": features,
        "target": "hbr_y9",
        "transform": "log10",
        "inverse": "pow10",
        "training_data": {
            "source": "MINTelligence malaria_simulations_4096.duckdb + HBR_malaria_simulations_4096.duckdb",
            "n_rows": len(DT),
            "n_params": DT["parameter_index"].nunique()
        },
        "cv": {
            "K": K_FOLDS,
            "stratify_by": f"strat_bin (k-means on log10(HBR), centers={K_STRATA})",
            "best_iteration_median": best_nrounds
        },
    }

    model_bundle = {
        "class": "estiMINT_EIR_to_HBR_model",
        "booster": xgb_final,
        "calibrator": cal_bundle,
        "features": features,
        "best_nrounds": best_nrounds,
        "preprocess": preprocess,
    }

    with open(dir_models / "estiMINT_EIR_to_HBR_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {dir_models}/estiMINT_EIR_to_HBR_model.pkl")
    print(f"Plots saved to: {dir_plots}/")
    print(f"Metrics saved to: {dir_metric}/")

    return 0

if __name__ == "__main__":
    sys.exit(main())
