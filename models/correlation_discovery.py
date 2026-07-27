"""Correlation Discovery

Three sources of findings, written to models/artifacts/correlations.json:
  - cross_correlation    lagged Pearson correlation across the 8 process variables,
                          computed within each transition so a lag never reads into
                          the next transition
  - model_importance     feature importances restricted to engineered
                          (rolling/roc/lag) features -- raw recipe inputs aren't a
                          "new" correlation, a rolling std the model leans on is
  - scenario_deviation    mean/max deviation_percent per transition_scenario

Obvious recipe relationships (stock<->basis_weight, steam<->moisture, filler<->ash)
are excluded from cross_correlation -- they're already in the recipe, not a find.
"""

import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/artifacts/risk_model.pkl")
OUTPUT_PATH = Path("models/artifacts/correlations.json")

PROCESS_VARS = [
    "basis_weight", "stock_flow", "filler_flow", "machine_speed",
    "steam_pressure", "moisture", "ash_content", "dryer_temp",
]
DESPIKE_VARS = ["basis_weight", "moisture"]
LAGS = [10, 30, 60]
CORR_THRESHOLD = 0.5
MIN_OVERLAP = 100
TOP_N_IMPORTANCE = 10
GROUP_COL = "transition_id"
SCENARIO_COL = "transition_scenario"

KNOWN_RECIPE_PAIRS = {
    frozenset({"stock_flow", "basis_weight"}),
    frozenset({"steam_pressure", "moisture"}),
    frozenset({"filler_flow", "ash_content"}),
}


def despike(series, window=7, n_sigmas=5.0):
    k = 1.4826
    rolling_median = series.rolling(window, center=True, min_periods=1).median()
    residual = (series - rolling_median).abs()
    mad = residual.rolling(window, center=True, min_periods=1).median()
    threshold = (n_sigmas * k * mad).replace(0, np.nan)
    is_spike = (residual > threshold).fillna(False)
    return series.where(~is_spike, rolling_median)


def load_data(path=FEATURES_PATH):
    df = pd.read_csv(path)
    for var in DESPIKE_VARS:
        df[var] = despike(df[var])
    return df


def scenario_per_transition(df):
    transitioning = df[df[SCENARIO_COL] != "SteadyState"]
    return transitioning.groupby(GROUP_COL)[SCENARIO_COL].first()


def lagged_correlation(df, current_var, past_var, lag):
    """corr(current_var_t, past_var_{t-lag}): does past_var's earlier value line
    up with current_var now. Shifted within transition_id so it can't cross into
    the next transition."""
    lagged = df.groupby(GROUP_COL)[past_var].shift(lag)
    valid = lagged.notna()
    if valid.sum() < MIN_OVERLAP:
        return np.nan
    return np.corrcoef(df.loc[valid, current_var], lagged[valid])[0, 1]


def cross_correlation_findings(df):
    findings = []
    for var_a, var_b in combinations(PROCESS_VARS, 2):
        if frozenset({var_a, var_b}) in KNOWN_RECIPE_PAIRS:
            continue
        for lag in LAGS:
            for current_var, past_var in [(var_a, var_b), (var_b, var_a)]:
                corr = lagged_correlation(df, current_var, past_var, lag)
                if pd.notna(corr) and abs(corr) > CORR_THRESHOLD:
                    findings.append({
                        "variable_a": past_var,
                        "variable_b": current_var,
                        "lag": lag,
                        "correlation": round(float(corr), 3),
                        "note": f"{past_var} {lag}s earlier tracks {current_var} now",
                        "source": "cross_correlation",
                    })
    findings.sort(key=lambda f: -abs(f["correlation"]))
    return findings


def is_engineered(feature_name):
    return "_roll_" in feature_name or feature_name.endswith("_roc") or "_lag" in feature_name


def model_importance_findings(model_path=MODEL_PATH, top_n=TOP_N_IMPORTANCE):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    ranked = sorted(
        zip(model.feature_names_in_, model.feature_importances_),
        key=lambda item: -item[1],
    )
    findings = [
        {
            "variable_a": name,
            "variable_b": "off_spec_risk",
            "lag": 0,
            "correlation": round(float(importance), 4),
            "note": "risk-model feature importance, not raw input the recipe defines",
            "source": "model_importance",
        }
        for name, importance in ranked
        if is_engineered(name)
    ][:top_n]
    return findings


def scenario_deviation_summary(df, scenario_map):
    scenario = df[GROUP_COL].map(scenario_map)
    summary = df.groupby(scenario)["deviation_percent"].agg(["mean", "max"]).reset_index()
    summary.columns = ["transition_scenario", "mean_deviation_pct", "max_deviation_pct"]
    return summary.round(2).to_dict(orient="records")


def main():
    df = load_data()
    scenario_map = scenario_per_transition(df)

    output = {
        "cross_correlation": cross_correlation_findings(df),
        "model_importance": model_importance_findings(),
        "scenario_deviation_summary": scenario_deviation_summary(df, scenario_map),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"cross_correlation findings: {len(output['cross_correlation'])}")
    print(f"model_importance findings: {len(output['model_importance'])}")
    print(json.dumps(output["scenario_deviation_summary"], indent=2))


if __name__ == "__main__":
    main()
