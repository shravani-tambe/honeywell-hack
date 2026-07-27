"""GET /api/trend?end_index={n}

Returns a sliding window of basis-weight, moisture, ash, setpoint, control
band, and a short linear extrapolation for the TrendView chart.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"

router = APIRouter()

_df = None


def _load():
    global _df
    if _df is None:
        if not FEATURES_PATH.exists():
            raise RuntimeError(f"features.csv not found at {FEATURES_PATH}. Run feature_engineering.py first.")
        _df = pd.read_csv(FEATURES_PATH)
    return _df


WINDOW = 120          # how many rows to send to the chart
EXTRAPOLATION = 15    # how many steps of linear projection


@router.get("/trend")
def get_trend(end_index: int = 700):
    df = _load()

    if end_index < 0 or end_index >= len(df):
        raise HTTPException(status_code=422, detail=f"end_index {end_index} out of range [0, {len(df) - 1}]")

    start = max(0, end_index - WINDOW + 1)
    window = df.iloc[start : end_index + 1]

    timestamps = window["timestamp"].tolist()
    basis_weight = window["basis_weight"].tolist()
    off_spec = window["off_spec"].astype(bool).tolist()
    moisture = window["moisture"].tolist()
    ash_content = window["ash_content"].tolist()

    setpoint = float(window["basis_weight_setpoint"].iloc[-1])
    band_pct = 0.025
    control_band = {"low": setpoint * (1 - band_pct), "high": setpoint * (1 + band_pct)}

    # linear extrapolation from last ~20 points
    tail = min(20, len(basis_weight))
    y_tail = np.array(basis_weight[-tail:])
    x_tail = np.arange(tail)
    coeffs = np.polyfit(x_tail, y_tail, 1)
    extrap = [float(np.polyval(coeffs, tail + i)) for i in range(EXTRAPOLATION)]

    return {
        "timestamps": timestamps,
        "basis_weight": basis_weight,
        "off_spec": off_spec,
        "moisture": moisture,
        "ash_content": ash_content,
        "basis_weight_setpoint": setpoint,
        "control_band": control_band,
        "extrapolation": {
            "start_timestamp": timestamps[-1] + 1 if timestamps else 0,
            "basis_weight": extrap,
        },
    }
