"""Predict child anaemia in places the models have never seen.

Experiment A: district hold-out, 5 folds of whole districts (the `fold` column).
Experiment B: leave-one-state-out, train on 35 states/UTs and test on the 36th.

Models: logistic regression (baseline) and LightGBM. The multilevel model's
out-of-fold predictions from R/multilevel.R join the comparison in experiment A.

Run from the project root (takes about 15-30 minutes):
    python src/models.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
SEED = 42
N_BOOT = 500

NUMERIC = [
    "child_age_months", "child_female", "birth_order", "stunted",
    "diarrhoea_2wk", "fever_2wk", "mother_age", "mother_edu", "mother_anaemic",
    "wealth_q", "rural", "clean_fuel", "improved_sanitation",
]
CATEGORICAL = ["ifa_pregnancy"]
FEATURES = NUMERIC + CATEGORICAL


def predict_out_of_fold(df, group_col):
    """Train on all groups but one, predict the held-out group, for every group."""
    logistic = make_pipeline(
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]),
        LogisticRegression(max_iter=2000),
    )
    gbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, verbose=-1,
    )
    X_gbm = df[FEATURES].astype({"ifa_pregnancy": "category"})
    y = df["anaemic"]
    preds = pd.DataFrame(index=df.index, columns=["p_logistic", "p_lightgbm"],
                         dtype=float)
    for g in sorted(df[group_col].unique()):
        test = (df[group_col] == g).to_numpy()
        logistic.fit(df.loc[~test, FEATURES], y[~test])
        preds.loc[test, "p_logistic"] = logistic.predict_proba(df.loc[test, FEATURES])[:, 1]
        gbm.fit(X_gbm[~test], y[~test])
        preds.loc[test, "p_lightgbm"] = gbm.predict_proba(X_gbm[test])[:, 1]
        print(f"  {group_col} = {g} done")
    return preds


def evaluate(y, p, districts, rng):
    """AUROC with a 95% CI from resampling whole districts, plus AUPRC and Brier."""
    y, p = np.asarray(y), np.asarray(p)
    rows_by_district = [np.flatnonzero(districts == d) for d in np.unique(districts)]
    boot = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(rows_by_district), len(rows_by_district))
        idx = np.concatenate([rows_by_district[i] for i in pick])
        boot.append(roc_auc_score(y[idx], p[idx]))
    return {
        "auroc": roc_auc_score(y, p),
        "auroc_ci_low": np.percentile(boot, 2.5),
        "auroc_ci_high": np.percentile(boot, 97.5),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "n_children": len(y),
    }


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(PROCESSED / "analysis.parquet")
    glmer = pd.read_csv(PROCESSED / "oof_glmer.csv")
    print(f"Children: {len(df):,}, anaemic: {df['anaemic'].mean():.1%} (unweighted)")

    print("\nExperiment A: unseen districts (5 district folds)")
    oof = predict_out_of_fold(df, "fold")
    oof = pd.concat([df[["row_id", "anaemic", "fold", "district_code", "state_code"]],
                     oof], axis=1).merge(glmer, on="row_id", how="left")

    results = []
    for model in ["p_logistic", "p_lightgbm"]:
        r = evaluate(oof["anaemic"], oof[model], oof["district_code"].to_numpy(), rng)
        results.append({"experiment": "A: unseen districts, all children",
                        "model": model[2:], **r})
    # Same children for all three models: those the multilevel model could use.
    both = oof.dropna(subset=["p_glmer"])
    for model in ["p_logistic", "p_lightgbm", "p_glmer"]:
        r = evaluate(both["anaemic"], both[model], both["district_code"].to_numpy(), rng)
        results.append({"experiment": "A: unseen districts, complete cases",
                        "model": model[2:], **r})

    print("\nExperiment B: unseen states (leave one state out)")
    loso = predict_out_of_fold(df, "state_code")
    for model in ["p_logistic", "p_lightgbm"]:
        r = evaluate(df["anaemic"], loso[model], df["district_code"].to_numpy(), rng)
        results.append({"experiment": "B: unseen states, all children",
                        "model": model[2:], **r})

    per_state = (pd.concat([df[["state_name", "anaemic"]], loso], axis=1)
                 .groupby("state_name")
                 .apply(lambda s: pd.Series({
                     "n_children": len(s),
                     "prevalence": s["anaemic"].mean(),
                     "auroc_logistic": roc_auc_score(s["anaemic"], s["p_logistic"]),
                     "auroc_lightgbm": roc_auc_score(s["anaemic"], s["p_lightgbm"]),
                 }), include_groups=False)
                 .sort_values("auroc_lightgbm"))

    # Calibration: mean predicted vs observed risk in deciles of predicted risk.
    calibration = []
    for model in ["p_logistic", "p_lightgbm", "p_glmer"]:
        d = oof.dropna(subset=[model])
        decile = pd.qcut(d[model], 10, labels=False, duplicates="drop")
        cal = d.groupby(decile).agg(predicted=(model, "mean"), observed=("anaemic", "mean"))
        calibration.append(cal.assign(model=model[2:]).reset_index(names="decile"))
    calibration = pd.concat(calibration)

    results = pd.DataFrame(results)
    TABLES.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(TABLES / "model_comparison.csv", index=False)
    per_state.round(4).to_csv(TABLES / "loso_per_state_auroc.csv")
    calibration.round(4).to_csv(TABLES / "calibration_unseen_districts.csv", index=False)
    oof.to_parquet(PROCESSED / "oof_predictions.parquet", index=False)

    pd.set_option("display.width", 140)
    print("\nModel comparison\n" + results.round(3).to_string(index=False))
    print("\nHardest 5 states to predict (LightGBM AUROC)\n"
          + per_state.head(5).round(3).to_string())
    print("\nSaved model_comparison.csv, loso_per_state_auroc.csv, "
          "calibration_unseen_districts.csv, oof_predictions.parquet")


if __name__ == "__main__":
    main()