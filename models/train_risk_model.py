"""Off-spec risk model.

Binary classifier: will basis weight go off-spec (>2.5% dev.) in the next
RISK_HORIZON seconds. Trained on data/processed/features.csv.

Label note: feature_engineering.py's off_spec_future (30s horizon, global
rolling window) is degenerate on this dataset -- off-spec streaks are so
frequent that no 30s window is ever fully in-spec, so every row ends up
labeled 1. 
It also rolls across transition boundaries near the tail of
each block. Script rebuilds the label with a shorter horizon,
computed per transition so no lookahead crosses into the next transition.
See build_target() below.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

FEATURES_PATH = Path("data/processed/features.csv")
ARTIFACT_DIR = Path("models/artifacts")
MODEL_PATH = ARTIFACT_DIR / "risk_model.pkl"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importances.json"
REPORT_PATH = ARTIFACT_DIR / "evaluation_report.json"

GROUP_COL = "transition_id"
SCENARIO_COL = "transition_scenario"
TARGET = "off_spec_risk"
RISK_HORIZON = 5
TEST_FRACTION = 0.2
VAL_FRACTION = 0.15
RANDOM_STATE = 42

LEAKAGE_COLS = ["timestamp", "off_spec_future", GROUP_COL, SCENARIO_COL]


def load_data(path=FEATURES_PATH):
    return pd.read_csv(path)


def build_target(df, horizon=RISK_HORIZON):
    """1 if off_spec occurs at any point in the next `horizon` steps, computed
    within each transition so the window never reads into the next one."""

    def forward_max(group):
        reversed_series = group[::-1]
        rolled = reversed_series.rolling(horizon, min_periods=horizon).max()[::-1]
        return rolled.shift(-1)

    return df.groupby(GROUP_COL)["off_spec"].transform(forward_max)


def scenario_per_transition(df):
    """transition_scenario is 'SteadyState' outside the transition window and
    the real scenario name during it; take the non-SteadyState label."""
    transitioning = df[df[SCENARIO_COL] != "SteadyState"]
    return transitioning.groupby(GROUP_COL)[SCENARIO_COL].first()


def build_feature_frame(df):
    X = df.drop(columns=LEAKAGE_COLS + [TARGET])
    X = pd.get_dummies(X, columns=["grade"], drop_first=True)
    X["is_transitioning"] = X["is_transitioning"].astype(int)
    return X


def stratified_group_split(transition_ids, scenario_map, fraction, rng):
    holdout = []
    for scenario in scenario_map.unique():
        group_ids = transition_ids[scenario_map.loc[transition_ids].values == scenario]
        group_ids = rng.permutation(group_ids)
        n_holdout = max(1, round(len(group_ids) * fraction))
        holdout.extend(group_ids[:n_holdout])
    return np.array(holdout)


def split_data(df, scenario_map):
    rng = np.random.default_rng(RANDOM_STATE)
    all_ids = scenario_map.index.to_numpy()

    test_ids = stratified_group_split(all_ids, scenario_map, TEST_FRACTION, rng)
    remaining_ids = np.setdiff1d(all_ids, test_ids)
    val_ids = stratified_group_split(remaining_ids, scenario_map, VAL_FRACTION, rng)
    train_ids = np.setdiff1d(remaining_ids, val_ids)

    groups = df[GROUP_COL]
    return (
        df[groups.isin(train_ids)],
        df[groups.isin(val_ids)],
        df[groups.isin(test_ids)],
    )


def train_model(X_train, y_train, X_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "n_test": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
    }


def evaluate_persistence_baseline(df_test, y_test):
    """Naive baseline: assume the next `horizon` seconds look like right now
    (predict off_spec_risk = current off_spec). The trained model should beat
    this, since it has genuine lead-time signal to add."""
    y_pred = df_test["off_spec"].values
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def evaluate_by_scenario(model, df_test, X_test, y_test, scenario_map):
    y_pred = model.predict(X_test)
    scenario = df_test[GROUP_COL].map(scenario_map).values
    results = {}
    for name in pd.unique(scenario):
        mask = scenario == name
        if mask.sum() == 0:
            continue
        results[name] = {
            "n": int(mask.sum()),
            "accuracy": accuracy_score(y_test[mask], y_pred[mask]),
            "f1": f1_score(y_test[mask], y_pred[mask], zero_division=0),
            "positive_rate": float(y_test[mask].mean()),
        }
    return results


def save_artifacts(model, feature_names, overall_metrics, scenario_metrics, baseline_metrics):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    importances = dict(
        sorted(
            zip(feature_names, model.feature_importances_.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    with open(IMPORTANCE_PATH, "w") as f:
        json.dump(importances, f, indent=2)

    report = {
        "target": TARGET,
        "risk_horizon_seconds": RISK_HORIZON,
        "overall": overall_metrics,
        "persistence_baseline": baseline_metrics,
        "by_transition_scenario": scenario_metrics,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)


def main():
    df = load_data()
    df[TARGET] = build_target(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    scenario_map = scenario_per_transition(df)
    train_df, val_df, test_df = split_data(df, scenario_map)

    X_train, y_train = build_feature_frame(train_df), train_df[TARGET]
    X_val, y_val = build_feature_frame(val_df), val_df[TARGET]
    X_test, y_test = build_feature_frame(test_df), test_df[TARGET]

    model = train_model(X_train, y_train, X_val, y_val)

    overall_metrics = evaluate(model, X_test, y_test)
    baseline_metrics = evaluate_persistence_baseline(test_df, y_test)
    scenario_metrics = evaluate_by_scenario(model, test_df, X_test, y_test, scenario_map)
    save_artifacts(model, X_train.columns.tolist(), overall_metrics, scenario_metrics, baseline_metrics)

    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print("model   ", json.dumps(overall_metrics, indent=2))
    print("baseline", json.dumps(baseline_metrics, indent=2))
    print(json.dumps(scenario_metrics, indent=2))


if __name__ == "__main__":
    main()
