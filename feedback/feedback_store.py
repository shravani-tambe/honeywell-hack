"""Feedback Store

Logs operator accept/reject decisions against suggestions and backfills
each suggestion's actual_outcome from the real trajectory that followed it, so
GET /api/accuracy has a measured number instead of an unverified prediction.
Storage is a flat CSV keyed by suggestion_id, per the roadmap's scope.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FEATURES_PATH = Path("data/processed/features.csv")
RECOMMENDATIONS_PATH = Path("models/artifacts/recommendations.json")
FEEDBACK_LOG_PATH = Path("feedback/feedback_log.csv")

RISK_HORIZON = 5  # must match train_risk_model.RISK_HORIZON

COLUMNS = [
    "suggestion_id", "transition_id", "timestamp", "variable",
    "recommended_value", "predicted_risk_reduction", "source",
    "user_decision", "decided_at", "actual_outcome", "outcome_backfilled_at",
]
DECISIONS = {"accepted", "rejected"}


def load_log(path=FEEDBACK_LOG_PATH):
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"suggestion_id": str}, keep_default_na=False)


def save_log(df, path=FEEDBACK_LOG_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def seed_from_recommendations(recommendations_path=RECOMMENDATIONS_PATH, path=FEEDBACK_LOG_PATH):
    """Add one pending row per suggestion not already logged. Safe to re-run."""
    suggestions = json.load(open(recommendations_path))
    df = load_log(path)
    known_ids = set(df["suggestion_id"])
    new_rows = [
        {
            "suggestion_id": s["suggestion_id"],
            "transition_id": s["transition_id"],
            "timestamp": s["timestamp"],
            "variable": s["variable"],
            "recommended_value": s["recommended_value"],
            "predicted_risk_reduction": s["predicted_risk_reduction"],
            "source": s["source"],
            "user_decision": "pending",
            "decided_at": "",
            "actual_outcome": "",
            "outcome_backfilled_at": "",
        }
        for s in suggestions
        if s["suggestion_id"] not in known_ids
    ]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_log(df, path)
    return df


def record_decision(suggestion_id, decision, path=FEEDBACK_LOG_PATH):
    """Log an operator's accept/reject response for one suggestion."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    df = load_log(path)
    mask = df["suggestion_id"] == suggestion_id
    if not mask.any():
        raise KeyError(f"unknown suggestion_id: {suggestion_id}")
    df.loc[mask, "user_decision"] = decision
    df.loc[mask, "decided_at"] = datetime.now(timezone.utc).isoformat()
    save_log(df, path)
    return df.loc[mask].iloc[0].to_dict()


def _transition_windows(features_path, columns=("timestamp", "transition_id", "off_spec")):
    features = pd.read_csv(features_path, usecols=list(columns))
    return {tid: g.sort_values("timestamp") for tid, g in features.groupby("transition_id")}


def _ground_truth_outcome(window, timestamp, horizon):
    """1 if off_spec occurs anywhere in the horizon seconds after timestamp,
    within the same transition -- the same forward-looking definition
    train_risk_model.build_target() uses, evaluated against what actually
    happened rather than against the model's prediction."""
    future = window[(window["timestamp"] > timestamp) & (window["timestamp"] <= timestamp + horizon)]
    if len(future) < horizon:
        return ""
    return "off_spec" if future["off_spec"].max() == 1 else "in_spec"


def backfill_actual_outcomes(df, features_path=FEATURES_PATH, horizon=RISK_HORIZON):
    """Fill actual_outcome for rows that don't have one yet. Reflects the
    trajectory that actually occurred, not a counterfactual replay of the
    recommended setpoint -- that requires the Phase 7 closed-loop simulator."""
    pending = df.index[df["actual_outcome"] == ""]
    if pending.empty:
        return df
    windows = _transition_windows(features_path)
    now = datetime.now(timezone.utc).isoformat()
    for idx in pending:
        row = df.loc[idx]
        window = windows.get(row["transition_id"])
        outcome = _ground_truth_outcome(window, row["timestamp"], horizon) if window is not None else ""
        if outcome:
            df.loc[idx, "actual_outcome"] = outcome
            df.loc[idx, "outcome_backfilled_at"] = now
    return df


def compute_accuracy(df=None, path=FEEDBACK_LOG_PATH, group_by=None):
    """Share of evaluated suggestions after which the process was actually
    in spec. Only counts rows with a resolved actual_outcome."""
    df = df if df is not None else load_log(path)
    evaluated = df[df["actual_outcome"] != ""]
    if evaluated.empty:
        return {"n_evaluated": 0, "accuracy": None}
    result = {
        "n_evaluated": int(len(evaluated)),
        "accuracy": round(float((evaluated["actual_outcome"] == "in_spec").mean()), 4),
    }
    if group_by:
        result[f"by_{group_by}"] = (
            evaluated.groupby(group_by)["actual_outcome"]
            .apply(lambda s: round(float((s == "in_spec").mean()), 4))
            .to_dict()
        )
    return result


def main():
    df = seed_from_recommendations()
    df = backfill_actual_outcomes(df)
    save_log(df)

    print(f"logged={len(df)}  pending_decisions={(df['user_decision'] == 'pending').sum()}")
    print(json.dumps(compute_accuracy(df, group_by="source"), indent=2))


if __name__ == "__main__":
    main()
