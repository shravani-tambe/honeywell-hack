"""GET /api/correlations

Returns the full correlations.json artifact produced by
correlation_discovery.py — cross-correlation findings, model importance,
and scenario deviation summary.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORRELATIONS_PATH = ROOT / "models" / "artifacts" / "correlations.json"

router = APIRouter()

_data = None


def _load():
    global _data
    if _data is None:
        if not CORRELATIONS_PATH.exists():
            raise HTTPException(status_code=404, detail="correlations.json not found. Run correlation_discovery.py first.")
        with open(CORRELATIONS_PATH) as f:
            _data = json.load(f)
    return _data


@router.get("/correlations")
def get_correlations():
    return _load()
