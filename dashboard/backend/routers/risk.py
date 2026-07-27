"""GET /api/risk?end_index={n}

Returns the off-spec risk probability at a given row index, predicted by the
XGBoost model, plus contextual metadata for the RiskPanel.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = ROOT / "models" / "artifacts" / "risk_model.pkl"

router = APIRouter()

_df = None
_model = None


def _load_data():
    global _df
    if _df is None:
        _df = pd.read_csv(FEATURES_PATH)
    return _df


def _load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def _build_feature_row(row_df, feature_names):
    """Replicate the feature matrix build from setpoint_recommender."""
    X = row_df.copy()
    X["grade_Grade_B"] = (X["grade"] == "Grade_B").astype(int)
    X["grade_Grade_C"] = (X["grade"] == "Grade_C").astype(int)
    X["is_transitioning"] = X["is_transitioning"].astype(int)
    return X[feature_names]


@router.get("/risk")
def get_risk(end_index: int = 700):
    df = _load_data()
    model = _load_model()

    if end_index < 0 or end_index >= len(df):
        raise HTTPException(status_code=422, detail=f"end_index {end_index} out of range [0, {len(df) - 1}]")

    row = df.iloc[[end_index]]
    feature_names = list(model.feature_names_in_)
    X = _build_feature_row(row, feature_names)
    risk_proba = float(model.predict_proba(X)[:, 1][0])

    row_data = df.iloc[end_index]
    return {
        "risk": risk_proba,
        "grade": str(row_data["grade"]),
        "transition_id": int(row_data["transition_id"]),
        "is_transitioning": bool(row_data["is_transitioning"]),
        "off_spec_now": bool(row_data["off_spec"]),
    }
