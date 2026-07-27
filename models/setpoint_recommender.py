"""Setpoint recommendation engine.

For process states at meaningful off-spec risk, perturbs each manipulated
variable by a small delta, reruns the risk model, and recommends
whichever perturbation cuts predicted risk the most. Recommendations are
clipped to recipe-derived safe bounds and tagged with a source of inference:
recipe_limit, correlation_discovery, or model_sensitivity.
"""

import json
import pickle
import uuid
from pathlib import Path

import pandas as pd

from correlation_discovery import CORR_THRESHOLD
from rationale_templates import build_rationale

FEATURES_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/artifacts/risk_model.pkl")
RECIPES_PATH = Path("data/recipes/grade_recipes.json")
CORRELATIONS_PATH = Path("models/artifacts/correlations.json")
OUTPUT_PATH = Path("models/artifacts/recommendations.json")

MANIPULATED_VARS = ["stock_flow", "filler_flow", "machine_speed", "steam_pressure"]
RECIPE_KEYS = {"stock_flow": "stock_flow", "filler_flow": "filler", "machine_speed": "speed", "steam_pressure": "steam"}
DELTA_FRACTION = 0.03
RISK_BAND = (0.15, 0.95)  # below: not enough risk to act on: above: horizon likely already lost
MIN_RISK_REDUCTION = 0.01
MIN_ACTIONABLE_DELTA = 0.005  # skip suggestions a recipe clip shrinks to a negligible change
MAX_PER_TRANSITION = 3
RANDOM_STATE = 42

SOURCE_RECIPE_LIMIT = "recipe_limit"
SOURCE_CORRELATION = "correlation_discovery"
SOURCE_SENSITIVITY = "model_sensitivity"


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_recipes(path=RECIPES_PATH):
    recipes = json.load(open(path))
    recipes.pop("_meta", None)
    return recipes


def load_correlations(path=CORRELATIONS_PATH):
    if not path.exists():
        return []
    return json.load(open(path))["cross_correlation"]


def recipe_bounds(recipes):
    return {
        var: (min(g[key] for g in recipes.values()), max(g[key] for g in recipes.values()))
        for var, key in RECIPE_KEYS.items()
    }


def build_feature_matrix(rows_df, feature_names):
    X = rows_df.copy()
    X["grade_Grade_B"] = (X["grade"] == "Grade_B").astype(int)
    X["grade_Grade_C"] = (X["grade"] == "Grade_C").astype(int)
    X["is_transitioning"] = X["is_transitioning"].astype(int)
    return X[feature_names]


def select_candidates(df, model, feature_names):
    """Rows worth suggesting for: mid-transition, with risk high enough to
    matter but not already certain, so an intervention has room to help."""
    trans = df[df["is_transitioning"]].copy()
    trans["baseline_risk"] = model.predict_proba(build_feature_matrix(trans, feature_names))[:, 1]
    low, high = RISK_BAND
    trans = trans[trans["baseline_risk"].between(low, high)]

    sampled = [
        group.sample(n=min(MAX_PER_TRANSITION, len(group)), random_state=RANDOM_STATE)
        for _, group in trans.groupby("transition_id")
    ]
    return pd.concat(sampled) if sampled else trans


def perturbation_batch(row):
    variants, labels = [row], [(None, None)]
    for var in MANIPULATED_VARS:
        current = row[var]
        for direction in (1, -1):
            perturbed = row.copy()
            perturbed[var] = current + direction * DELTA_FRACTION * current
            variants.append(perturbed)
            labels.append((var, direction))
    return labels, pd.DataFrame(variants)


def best_perturbation(model, feature_names, row):
    labels, variants = perturbation_batch(row)
    proba = model.predict_proba(build_feature_matrix(variants, feature_names))[:, 1]
    baseline = proba[0]

    best = None
    for (var, direction), risk in zip(labels[1:], proba[1:]):
        reduction = baseline - risk
        if best is None or reduction > best["reduction"]:
            current = row[var]
            best = {
                "variable": var,
                "candidate": current + direction * DELTA_FRACTION * current,
                "reduction": reduction,
            }
    return baseline, best


def risk_at(model, feature_names, row, variable, value):
    variant = row.copy()
    variant[variable] = value
    proba = model.predict_proba(build_feature_matrix(pd.DataFrame([variant]), feature_names))[:, 1]
    return float(proba[0])


def find_supporting_correlation(variable, correlations):
    matches = [
        c for c in correlations
        if c["variable_a"] == variable
        and c["variable_b"] == "basis_weight"
        and abs(c["correlation"]) >= CORR_THRESHOLD
    ]
    return max(matches, key=lambda c: abs(c["correlation"])) if matches else None


def clip_to_recipe(variable, candidate, bounds):
    low, high = bounds[variable]
    clipped = max(low, min(high, candidate))
    return clipped, clipped != candidate


def generate_suggestion(row, model, feature_names, bounds, correlations):
    baseline, best = best_perturbation(model, feature_names, row)
    if best is None or best["reduction"] < MIN_RISK_REDUCTION:
        return None

    variable = best["variable"]
    current = row[variable]
    recommended, was_clipped = clip_to_recipe(variable, best["candidate"], bounds)

    if abs(recommended - current) / current < MIN_ACTIONABLE_DELTA:
        return None

    reduction = baseline - risk_at(model, feature_names, row, variable, recommended) if was_clipped else best["reduction"]
    if reduction < MIN_RISK_REDUCTION:
        return None

    correlation = None if was_clipped else find_supporting_correlation(variable, correlations)
    source = SOURCE_RECIPE_LIMIT if was_clipped else (SOURCE_CORRELATION if correlation else SOURCE_SENSITIVITY)

    return {
        "suggestion_id": str(uuid.uuid4()),
        "timestamp": float(row["timestamp"]),
        "transition_id": int(row["transition_id"]),
        "variable": variable,
        "current_value": round(float(current), 2),
        "recommended_value": round(float(recommended), 2),
        "predicted_risk_reduction": round(float(reduction), 4),
        "rationale": build_rationale(
            variable, current, recommended, reduction, row["bw_error_pct"],
            correlation=correlation, clipped=was_clipped,
        ),
        "source": source,
        "status": "pending",
    }


def generate_suggestions(candidates, model, feature_names, bounds, correlations):
    suggestions = [
        generate_suggestion(row, model, feature_names, bounds, correlations)
        for _, row in candidates.iterrows()
    ]
    return [s for s in suggestions if s is not None]


def main():
    df = pd.read_csv(FEATURES_PATH)
    model = load_model()
    feature_names = list(model.feature_names_in_)
    recipes = load_recipes()
    bounds = recipe_bounds(recipes)
    correlations = load_correlations()

    candidates = select_candidates(df, model, feature_names)
    suggestions = generate_suggestions(candidates, model, feature_names, bounds, correlations)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(suggestions, f, indent=2)

    by_source = pd.Series([s["source"] for s in suggestions]).value_counts().to_dict()
    print(f"candidates={len(candidates)}  suggestions={len(suggestions)}")
    print(f"by source: {by_source}")


if __name__ == "__main__":
    main()
